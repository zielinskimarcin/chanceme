"""Private durable request journal. Never stores codes, PDFs, scores or IPs."""
from __future__ import annotations
import hashlib
import json
import os
import re
import time

PREFIX = 'chanceme:report-journal:v1:'
EVENT_STATUS = {'code_submitted':'waiting_for_code','code_send_error':'code_send_error',
                'email_verified':'email_verified','report_submitted':'report_submitted',
                'report_send_error':'report_send_error','code_locked':'code_locked'}
EVENTS = set(EVENT_STATUS) | {'code_rejected'}


def record_id(ident):
    return hashlib.sha256(ident.encode()).hexdigest()


def keys(ident):
    return PREFIX + record_id(ident), PREFIX + 'index'


def new_record(ident, email, test_request, source='request'):
    now = time.time()
    return dict(id=record_id(ident), email=email, created_at=now, updated_at=now,
                code_expires_at=now+900, status='requested', attempts=0,
                environment=os.environ.get('VERCEL_ENV', 'development'),
                test_request=bool(test_request), source=source,
                events=[dict(event='requested' if source=='request' else 'tracking_started', at=now)])


class Journal:
    def __init__(self, command):
        self.command = command

    def event(self, ident, kind, provider_id=None, email=None, test_request=False, attempts=None):
        if kind not in EVENTS:
            raise ValueError('Unknown journal event')
        now=time.time()
        data=dict(kind=kind, at=now, status=EVENT_STATUS.get(kind))
        if provider_id is not None:
            if not isinstance(provider_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',provider_id):
                raise ValueError('Invalid provider reference')
            data['provider_id']=provider_id
        if attempts is not None:
            data['attempts']=int(attempts)
        # Support valid requests already in flight at deployment without inventing old timestamps.
        fallback=new_record(ident,email,test_request,'inflight_at_release') if email else None
        lua='''
        local raw=redis.call('GET',KEYS[1])
        if not raw and ARGV[2]=='null' then return 0 end
        local p=cjson.decode(raw or ARGV[2]); local e=cjson.decode(ARGV[1])
        local field=e.kind..'_at'
        if not p[field] then
          p[field]=e.at
          table.insert(p.events,{event=e.kind,at=e.at})
          if #p.events>32 then table.remove(p.events,1) end
        end
        if e.status~=cjson.null and p.status~='report_submitted' then p.status=e.status end
        if e.attempts then p.attempts=e.attempts end
        if e.provider_id then
          if e.kind=='code_submitted' then p.code_provider_id=e.provider_id end
          if e.kind=='report_submitted' then p.report_provider_id=e.provider_id end
        end
        p.updated_at=e.at
        redis.call('SET',KEYS[1],cjson.encode(p))
        redis.call('ZADD',KEYS[2],p.created_at,p.id)
        return 1
        '''
        return self.command('EVAL',lua,2,*keys(ident),json.dumps(data),json.dumps(fallback))

    def list(self, offset=0, limit=100):
        if not isinstance(offset,int) or offset<0 or not isinstance(limit,int) or not 1<=limit<=100:
            raise ValueError('Invalid journal page')
        ids=self.command('ZREVRANGE',PREFIX+'index',offset,offset+limit-1)
        if not ids:return []
        values=self.command('MGET',*[PREFIX+i for i in ids])
        return [json.loads(value) for value in values if value]

    def all(self):
        # Snapshot the index once so new arrivals cannot shift offset-based pagination.
        ids=self.command('ZREVRANGE',PREFIX+'index',0,-1)
        rows=[]
        for start in range(0,len(ids),100):
            values=self.command('MGET',*[PREFIX+i for i in ids[start:start+100]])
            rows.extend(json.loads(v) for v in values if v)
        return rows


def display_status(record, now=None):
    if record['status'] in ('requested','waiting_for_code') and (now or time.time())>=record['code_expires_at']:
        return 'expired'
    return record['status']
