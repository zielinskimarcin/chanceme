"""Minimal transactional emails using Chance Me's typography and palette."""
import re

FONT = "Nunito,Arial,Helvetica,sans-serif"


def layout(title, preview, content):
    # Only module-owned copy and validated codes enter the template.
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><title>{title}</title>
<style>@font-face{{font-family:Nunito;src:url('https://chance-me.com/assets/nunito-latin.woff2') format('woff2');font-weight:200 1000;font-style:normal}}@media only screen and (max-width:520px){{.outer{{padding:32px 24px!important}}}}</style></head>
<body style="margin:0;padding:0;background:#ffffff;color:#17324d;font-family:{FONT};-webkit-text-size-adjust:100%">
<div style="display:none;font-size:1px;line-height:1px;max-height:0;max-width:0;opacity:0;overflow:hidden;mso-hide:all">{preview}</div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"><tr><td class="outer" align="center" style="padding:48px 24px">
<!--[if mso]><table role="presentation" width="460"><tr><td><![endif]-->
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="max-width:460px;font-family:{FONT}">
<tr><td style="padding-bottom:32px"><a href="https://chance-me.com/" style="color:#17324d;text-decoration:none;font-family:{FONT};font-size:27px;line-height:32px;font-weight:900;letter-spacing:-1px">chance me</a></td></tr>
<tr><td>{content}</td></tr>
<tr><td style="padding-top:28px;font-family:{FONT};font-size:11px;line-height:18px;color:#687f92">Not affiliated with Bocconi University. <a href="https://chance-me.com/privacy/" style="color:#687f92;text-decoration:underline">Privacy</a></td></tr>
</table><!--[if mso]></td></tr></table><![endif]-->
</td></tr></table></body></html>'''


def confirmation_html(code):
    if not isinstance(code, str) or not re.fullmatch(r'[0-9]{6}', code):
        raise ValueError('A six-digit code is required')
    return layout('Confirm your email', 'Your code expires in 15 minutes.', f'''
<h1 style="margin:0 0 12px;font-family:{FONT};font-size:24px;line-height:32px;font-weight:800;color:#17324d">Confirm your email</h1>
<p style="margin:0 0 22px;font-size:15px;line-height:24px;color:#58718c">Enter this code on Chance Me to receive your report.</p>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#edf7f2;border-radius:10px"><tr><td style="padding:18px 22px;font-family:{FONT};font-size:34px;line-height:44px;font-weight:800;color:#17324d">{code}</td></tr></table>
<p style="margin:10px 0 24px;font-size:13px;line-height:21px;color:#58718c">Expires in 15 minutes.</p>
<p style="margin:0;font-size:12px;line-height:20px;color:#687f92">If you didn’t request this, ignore this email.</p>''')


def report_html():
    return layout('Your Chance Me report', 'Your PDF report is attached.', f'''
<h1 style="margin:0 0 12px;font-family:{FONT};font-size:24px;line-height:32px;font-weight:800;color:#17324d">Your report</h1>
<p style="margin:0 0 16px;font-size:15px;line-height:24px;color:#58718c">Your PDF is attached.</p>
<p style="margin:0;font-size:14px;line-height:22px;color:#17324d">chance-me-report.pdf</p>''')
