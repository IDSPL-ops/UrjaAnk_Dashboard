from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'data/processed/manifest.json').exists(),reason='Prepared dataset not checked out')

def test_every_insight_view_and_its_export():
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=120).run()
    assert not app.exception
    assert app.sidebar.radio[0].value=='0 · Data coverage'
    for view in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(view).run()
        assert not app.exception,view
        assert any(x.label=='Download this view: datasets + inferences (ZIP)' for x in app.get('download_button'))
    app.button[0].click().run() # selected cohort schedule
    assert not app.exception


def test_apartment_date_and_unsupported_cases():
    app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=120).run()
    app.sidebar.radio[0].set_value('1 · Apartment energy fingerprint').run()
    for apt in app.sidebar.selectbox[0].options:
        app.sidebar.selectbox[0].set_value(apt).run();assert not app.exception,apt
    app.sidebar.selectbox[0].set_value('Lakeside A 701').run()
    app.sidebar.date_input[0].set_value((date(2025,5,6),date(2025,5,8))).run();assert not app.exception
    for view in ['2 · Energy lifestyle / behavior clustering','3 · Weather-normalized consumption','4 · Cooling sensitivity index','9 · Flexible load / demand response','10 · Matched-apartment benchmark']:
        app.sidebar.radio[0].set_value(view).run();assert not app.exception,view
    app.sidebar.date_input[0].set_value((date(2025,5,2),date(2026,5,26))).run()
    app.sidebar.selectbox[0].set_value('Urabno Q606').run()
    assert not app.exception and any(x.label=='Actual / expected' for x in app.metric)
    app.sidebar.radio[0].set_value('4 · Cooling sensitivity index').run()
    app.button[0].click().run();assert not app.exception
    assert any(x.value=='Validated cooling sensitivity comparison' for x in app.subheader)


def test_indoor_views_with_isolated_sensor_fixture(tmp_path):
    # Synthetic sensor values exercise rendering only and never enter saved data.
    idx=pd.date_range('2025-06-01',periods=14*96,freq='15min')
    fixture=tmp_path/'indoor.parquet'
    pd.DataFrame({'Timestamp':idx,'Apartment':'Lakeside K 502','Room':'Test bedroom','temperature':28+np.sin(np.arange(len(idx))/96*2*np.pi),'humidity':65+np.cos(np.arange(len(idx))/96*2*np.pi)}).to_parquet(fixture)
    code=(ROOT/'streamlit_app.py').read_text().replace("ROOT=Path(__file__).resolve().parent",f"ROOT=Path({str(ROOT)!r})")
    code=code.replace("elif (DATA/'indoor.parquet').exists():indoor=validate_indoor(pd.read_parquet(DATA/'indoor.parquet'))",f"elif True:indoor=validate_indoor(pd.read_parquet({str(fixture)!r}))")
    app=AppTest.from_string(code,default_timeout=120).run()
    app.sidebar.radio[0].set_value('5 · Indoor–outdoor thermal buffering').run()
    main_select=[x for x in app.selectbox if x.label=='AC channel serving this room'][0]
    main_select.set_value(main_select.options[0]).run();assert not app.exception
    for view in ['6 · Apartment thermal time constant','7 · AC cooling effectiveness','8 · Humidity / latent cooling performance']:
        app.sidebar.radio[0].set_value(view).run()
        ac=[x for x in app.selectbox if x.label=='AC channel serving this room'][0]
        ac.set_value(ac.options[0]).run();assert not app.exception,view
