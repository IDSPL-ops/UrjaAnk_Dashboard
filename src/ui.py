import io,json,zipfile
import pandas as pd
import plotly.express as px
import streamlit as st
PALETTE=['#087F8C','#E99841','#8A6EB0','#5CA18A','#D66A6A','#8793A2','#476985']
EVIDENCE={}

def json_text(value):
    # JSON null, not non-standard NaN/Infinity, for unavailable statistics.
    safe=json.loads(json.dumps(value,default=str),parse_constant=lambda _:None)
    return json.dumps(safe,indent=2,allow_nan=False)

def download(df,name,key,label='Download dataset (CSV)'):
    st.download_button(label,df.to_csv(index=False).encode('utf-8-sig'),file_name=name+'.csv',mime='text/csv',key=key)

def chart(fig,data,title,inference,identity,context):
    st.subheader(title)
    fig.update_layout(template='plotly_white',height=650 if identity=='coverage_channel' else 345,margin=dict(l=15,r=15,t=15,b=20),font=dict(family='Arial',size=12,color='#152536'),paper_bgcolor='rgba(0,0,0,0)',legend_title_text='')
    fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor='#E7EDF2')
    if data.empty:
        st.info('No eligible observations are available for this view and reporting period.')
    else:st.plotly_chart(fig,width='stretch',key=identity,config={'displaylogo':False,'toImageButtonOptions':{'format':'png','filename':identity,'scale':3}})
    st.markdown('**Inference:** '+inference)
    evidence={'chart':title,'inference':inference,**context,'rows':len(data)}
    EVIDENCE[identity]=(data.copy(),evidence)
    c1,c2=st.columns([2,1])
    with c1:download(data,identity,'csv_'+identity)
    with c2:st.download_button('Download inference + method',json_text(evidence),file_name=identity+'_evidence.json',mime='application/json',key='meta_'+identity)

def evidence_zip(extra=None):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for name,(d,m) in EVIDENCE.items():
            z.writestr(name+'.csv',d.to_csv(index=False));z.writestr(name+'_evidence.json',json_text(m))
        for name,content in (extra or {}).items():z.writestr(name,content)
    return b.getvalue()

def empty(text):st.info(text)

def fmt(value,digits=1):return 'Unavailable' if pd.isna(value) else f'{value:,.{digits}f}'
