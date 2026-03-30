from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))

path_sat_api = os.path.join(current_dir, 'bocconi_master_dataset.csv')
path_sat_root = os.path.join(os.path.dirname(current_dir), 'bocconi_master_dataset.csv')
CSV_PATH_SAT = path_sat_api if os.path.exists(path_sat_api) else (path_sat_root if os.path.exists(path_sat_root) else 'bocconi_master_dataset.csv')

try:
    df_sat = pd.read_csv(CSV_PATH_SAT, sep=None, engine='python')
    df_sat.columns = df_sat.columns.str.strip()
    if 'Decision' in df_sat.columns:
        df_sat = df_sat[df_sat['Decision'] != 'unknown']
        df_sat = df_sat.dropna(subset=['SAT', 'GPA', 'Decision', 'Course'])
    else:
        df_sat = pd.DataFrame(columns=['SAT', 'GPA', 'Course', 'Decision', 'Academic_Year', 'Session'])
except Exception:
    df_sat = pd.DataFrame(columns=['SAT', 'GPA', 'Course', 'Decision', 'Academic_Year', 'Session'])

path_bt_api = os.path.join(current_dir, 'bocconitest.csv')
path_bt_root = os.path.join(os.path.dirname(current_dir), 'bocconitest.csv')
CSV_PATH_BT = path_bt_api if os.path.exists(path_bt_api) else (path_bt_root if os.path.exists(path_bt_root) else 'bocconitest.csv')

try:
    df_bt = pd.read_csv(CSV_PATH_BT, sep=';', engine='python')
    df_bt.columns = df_bt.columns.str.strip()
    if 'Decision' in df_bt.columns:
        df_bt = df_bt.dropna(subset=['Bocconi_Test', 'GPA', 'Decision', 'Course'])
    else:
        df_bt = pd.DataFrame(columns=['Bocconi_Test', 'GPA', 'Course', 'Decision', 'Academic_Year', 'Session', 'Bocconi_Score'])
except Exception:
    df_bt = pd.DataFrame(columns=['Bocconi_Test', 'GPA', 'Course', 'Decision', 'Academic_Year', 'Session', 'Bocconi_Score'])

def calculate_chances_sat(user_sat, user_gpa, course, user_session):
    results = {
        'Course': course, 
        'User_Stats': {'SAT': user_sat, 'GPA': user_gpa, 'Session': user_session}, 
        'Status': {}, 
        'Historical_Data': {}
    }
    
    df_recent = df_sat[(df_sat['Academic_Year'].isin(['25-26', '26-27'])) & (df_sat['Course'] == course)].copy()
    
    zone = "Unknown"
    reason = "Insufficient data for this course in the recent cycles."
    
    if not df_recent.empty:
        similar = df_recent[
            (df_recent['SAT'].between(user_sat - 30, user_sat + 30)) & 
            (df_recent['GPA'].between(user_gpa - 0.2, user_gpa + 0.2))
        ]
        
        total_similar = len(similar)
        if total_similar > 0:
            accepted_similar = len(similar[similar['Decision'] == 'accept'])
            acc_rate = accepted_similar / total_similar
            
            if acc_rate >= 0.75:
                zone = "Safe"
            elif acc_rate >= 0.15:
                zone = "Competitive"
            else:
                zone = "High Risk"
                
            reason = f"Based on {total_similar} strictly matching profiles from the last two cycles, the historical acceptance rate is {int(acc_rate*100)}%. (Note: The nearest neighbors table below may show rejections that fall just outside this exact radius)."
        else:
            worse_or_same = df_recent[(df_recent['SAT'] <= user_sat) & (df_recent['GPA'] <= user_gpa)]
            accepted_worse = len(worse_or_same[worse_or_same['Decision'] == 'accept'])
            
            if accepted_worse >= 5:
                zone = "Safe"
                reason = f"No strictly similar profiles found in the exact radius. However, {accepted_worse} applicants with equal or lower statistics were accepted in the last two cycles."
            elif accepted_worse >= 1:
                zone = "Competitive"
                reason = f"No strictly similar profiles found. Only {accepted_worse} applicant(s) with equal or lower statistics were accepted in the last two cycles."
            else:
                zone = "High Risk"
                reason = "No applicants with similar, equal, or lower statistics were accepted in the last two cycles."

    if user_sat >= 1540 and user_gpa >= 8.5 and zone == "High Risk":
        zone = "Competitive"
        reason = "Limited exact matches found for such high SAT scores, but your profile remains competitive for this program."

    if user_sat < 1380 and zone == "Safe":
        zone = "Competitive"
        reason = "While your GPA is strong, a SAT score below 1380 is statistically risky for current standards."

    # --- HARD THRESHOLD ---
    if user_sat < 1400 and user_gpa < 8.0:
        zone = "High Risk"
        reason = "Statistically, profiles with a SAT below 1400 combined with a GPA below 8.0 fall into the highest risk category, regardless of any isolated historical outliers."
    
    # --- SPRING SESSION ---
    elif user_session == 'Spring':
        zone = "High Risk"
        reason = "The Spring session allocates only ~5% of total spots. While standard metrics might suggest otherwise, it is highly competitive and unpredictable."

    results['Status']['Zone'] = zone
    results['Status']['Reason'] = reason

    w_gpa = 2.0
    w_sat = 1.0
    target_sat_info = None
    
    for year in ['26-27', '25-26', '24-25']:
        df_year = df_sat[(df_sat['Academic_Year'] == year) & (df_sat['Course'] == course)].copy()
        if df_year.empty:
            results['Historical_Data'][year] = []
            continue
            
        base_dist = np.sqrt(w_sat * ((df_year['SAT'] - user_sat)/600)**2 + w_gpa * ((df_year['GPA'] - user_gpa)/10)**2)
        session_penalty = np.where(df_year['Session'] == user_session, 0.0, 0.015)
        
        df_year['dist'] = base_dist + session_penalty
        df_year = df_year.sort_values('dist')
        
        top_8 = df_year.head(8).to_dict('records')
        closest_profiles = top_8[:5] 
        for row in top_8[5:]:
            if row['dist'] <= 0.085: 
                closest_profiles.append(row)
                
        results['Historical_Data'][year] = closest_profiles
        
        if year == '26-27' and results['Status'].get('Zone') in ['High Risk', 'Competitive']:
            df_acc = df_year[(df_year['Decision'] == 'accept') & 
                             (df_year['GPA'] >= user_gpa - 0.2) & 
                             (df_year['GPA'] <= user_gpa + 0.2) & 
                             (df_year['SAT'] > user_sat)]
            if not df_acc.empty:
                target = df_acc.sort_values('SAT').iloc[0]
                target_sat_info = f"To improve statistical probability with a GPA of {target['GPA']}, aim for a SAT score of ~{target['SAT']:.0f}."
            else:
                df_acc_w = df_year[(df_year['Decision'] == 'accept') & 
                                   (df_year['GPA'] >= user_gpa - 0.4) & 
                                   (df_year['GPA'] <= user_gpa + 0.4) & 
                                   (df_year['SAT'] > user_sat)]
                if not df_acc_w.empty:
                    target = df_acc_w.sort_values('SAT').iloc[0]
                    target_sat_info = f"No acceptances found with a closely matching GPA. The nearest accepted profile required a GPA of {target['GPA']} and a SAT of {target['SAT']:.0f}."
                else:
                    base_msg = ""
                    if user_gpa >= 10.0 and user_sat >= 1600:
                        base_msg = "You have maximum academic statistics. Focus entirely on your CV and motivational letter."
                    elif user_gpa >= 10.0:
                        base_msg = "Since your GPA is at the maximum, an improvement in your SAT score is the only statistical path to increase probability."
                    elif user_sat >= 1600:
                        base_msg = "Since your SAT is at the maximum, an improvement in your GPA is the only statistical path to increase probability."
                    else:
                        base_msg = "Data indicates that an improvement in both GPA and SAT is statistically necessary."
                    target_sat_info = base_msg
                    
    if zone in ["High Risk", "Competitive"] and target_sat_info:
        target_sat_info += " Please review the historical data below for more context."

    results['Status']['Target_Advice'] = target_sat_info
    return results

def calculate_chances_bt(user_bt, user_gpa, course, user_session):
    results = {
        'Course': course, 
        'User_Stats': {'SAT': user_bt, 'GPA': user_gpa, 'Session': user_session}, 
        'Status': {}, 
        'Historical_Data': {}
    }
    
    tier_1 = ['BIEF', 'BEMACS', 'BAI']
    tier_2 = ['BIEM', 'BESS']
    tier_3 = ['BIG', 'BGL', 'CLEAM', 'BEMACC', 'CLEACC', 'CLMG']
    
    if course in tier_1:
        course_tier = tier_1
    elif course in tier_2:
        course_tier = tier_2
    else:
        course_tier = tier_3

    df_tier = df_bt[df_bt['Course'].isin(course_tier)].copy()
    
    user_score = (user_bt / 50.0) * 0.55 + (user_gpa / 10.0) * 0.45
    
    zone = "Unknown"
    reason = ""

    if not df_tier.empty:
        similar = df_tier[
            (df_tier['Bocconi_Test'].between(user_bt - 2.5, user_bt + 2.5)) & 
            (df_tier['GPA'].between(user_gpa - 0.25, user_gpa + 0.25))
        ]
        total_similar = len(similar)
        
        if total_similar >= 3:
            accepted_similar = len(similar[similar['Decision'] == 'accept'])
            acc_rate = accepted_similar / total_similar
            
            if acc_rate >= 0.70:
                zone = "Safe"
            elif acc_rate >= 0.20:
                zone = "Competitive"
            else:
                zone = "High Risk"
            
            reason = f"Based on {total_similar} highly similar profiles from your program tier, the historical acceptance rate is {int(acc_rate*100)}%. (Note: Exact historical matches are shown in the tables below)."
        
        else:
            df_tier_acc = df_tier[df_tier['Decision'] == 'accept']
            if not df_tier_acc.empty:
                median_score = df_tier_acc['Bocconi_Score'].median()
                p25_score = df_tier_acc['Bocconi_Score'].quantile(0.25)
                
                if user_score >= median_score:
                    zone = "Safe"
                    reason = "Not enough exact historical matches found for your specific scores. However, based on the global statistical distribution, your overall Bocconi Score places you securely above the median of accepted students for this tier."
                elif user_score >= p25_score:
                    zone = "Competitive"
                    reason = "Not enough exact historical matches found. However, your overall Bocconi Score places you within the competitive range (above the 25th percentile) of accepted students for this tier."
                else:
                    zone = "High Risk"
                    reason = "Not enough exact historical matches found. Based on global distribution, your overall Bocconi Score falls below the 25th percentile of typically accepted students for this tier."
            else:
                zone = "High Risk"
                reason = "Insufficient historical acceptance data for this program tier to make a statistical prediction."

    if user_bt < 25 or user_gpa < 7.0:
        zone = "High Risk"
        reason = "Statistically, a Bocconi Test score below 25/50 or a GPA below 7.0 places you in the highest risk category, regardless of other metrics."
    
    elif user_session == 'Spring':
        if zone == "Safe":
            zone = "Competitive" 
        reason = "The Spring session allocates only ~5% of total spots. Even with a highly competitive Bocconi Test score, admissions during this round are highly unpredictable."

    results['Status']['Zone'] = zone
    results['Status']['Reason'] = reason

    w_gpa = 2.0
    w_bt = 1.0
    target_info = None
    
    for year in ['26-27', '25-26', '24-25']:
        df_year = df_bt[(df_bt['Academic_Year'] == year) & (df_bt['Course'] == course)].copy()
        
        if df_year.empty:
             df_year = df_bt[(df_bt['Academic_Year'] == year) & (df_bt['Course'].isin(course_tier))].copy()
             
        if df_year.empty:
            results['Historical_Data'][year] = []
            continue
            
        base_dist = np.sqrt(w_bt * ((df_year['Bocconi_Test'] - user_bt)/50)**2 + w_gpa * ((df_year['GPA'] - user_gpa)/10)**2)
        session_penalty = np.where(df_year['Session'] == user_session, 0.0, 0.02) 
        
        df_year['dist'] = base_dist + session_penalty
        df_year = df_year.sort_values('dist')
        
        df_year_formatted = df_year.rename(columns={'Bocconi_Test': 'SAT'})
        
        top_8 = df_year_formatted.head(8).to_dict('records')
        closest_profiles = top_8[:5] 
        for row in top_8[5:]:
            if row['dist'] <= 0.085: 
                closest_profiles.append(row)
                
        results['Historical_Data'][year] = closest_profiles
        
        if year == '26-27' and results['Status'].get('Zone') in ['High Risk', 'Competitive']:
            df_acc = df_year[(df_year['Decision'] == 'accept') & 
                             (df_year['GPA'] >= user_gpa - 0.25) & 
                             (df_year['GPA'] <= user_gpa + 0.25) & 
                             (df_year['Bocconi_Test'] > user_bt)]
            if not df_acc.empty:
                target = df_acc.sort_values('Bocconi_Test').iloc[0]
                target_info = f"To improve statistical probability with a GPA of {target['GPA']}, aim for a Bocconi Test score of ~{target['Bocconi_Test']:.1f}."
            else:
                target_info = "Data indicates that an improvement in both GPA and Bocconi Test score is statistically necessary."
                    
    if zone in ["High Risk", "Competitive"] and target_info:
        target_info += " Please review the historical data below for more context."

    results['Status']['Target_Advice'] = target_info
    return results

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    
    score = float(data.get('sat')) 
    gpa = float(data.get('gpa'))
    course = data.get('course')
    session = data.get('session', 'Winter')
    
    if score <= 50:
        result = calculate_chances_bt(score, gpa, course, session)
    else:
        result = calculate_chances_sat(score, gpa, course, session)
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)