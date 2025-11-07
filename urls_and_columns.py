# urls_and_columns.py

# --- 1. URLs de Entrenamiento (2007-2016) ---
# (Esta sección estaba correcta y no se modifica)
TRAIN_URLS = {
    "BPX": { # Label B
        "2007": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BPX_E.xpt",
        "2009": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BPX_F.xpt",
        "2011": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BPX_G.xpt",
        "2013": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPX_H.xpt",
        "2015": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.xpt",
    },
    "DEMO": { # Features
        "2007": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DEMO_E.xpt",
        "2009": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DEMO_F.xpt",
        "2011": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt",
        "2013": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.xpt",
        "2015": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt",
    },
    "BMX": { # Features
        "2007": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BMX_E.xpt",
        "2009": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BMX_F.xpt",
        "2011": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BMX_G.xpt",
        "2013": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BMX_H.xpt",
        "2015": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.xpt",
    },
    "PAQ": { # Features
        "2007": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/PAQ_E.xpt",
        "2009": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/PAQ_F.xpt",
        "2011": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAQ_G.xpt",
        "2013": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAQ_H.xpt",
        "2015": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/PAQ_I.xpt",
    },
    "SLQ": { # Features
        "2007": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/SLQ_E.xpt",
        "2009": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/SLQ_F.xpt",
        "2011": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SLQ_G.xpt",
        "2013": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SLQ_H.xpt",
        "2015": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.xpt",
    },
    "DBQ": { # Features
        "2007": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DBQ_E.xpt",
        "2009": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DBQ_F.xpt",
        "2011": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DBQ_G.xpt",
        "2013": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DBQ_H.xpt",
        "2015": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DBQ_I.xpt",
    },
    "BPQ": { # Features
        "2007": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BPQ_E.xpt",
        "2009": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BPQ_F.xpt",
        "2011": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BPQ_G.xpt",
        "2013": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPQ_H.xpt",
        "2015": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPQ_I.xpt",
    },
    "SMQ": { # Features
        "2007": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/SMQ_E.xpt",
        "2009": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/SMQ_F.xpt",
        "2011": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SMQ_G.xpt",
        "2013": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SMQ_H.xpt",
        "2015": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.xpt",
    }
}

# --- 2. URLs de Test Ciego (2017-Mar 2020) ---
# --- ESTE BLOQUE HA SIDO CORREGIDO con tus links ---
TEST_URLS = {
    "BPX": { # Label B
        "2017": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.xpt",
    },
    "DEMO": { # Features
        "2017": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.xpt",
    },
    "BMX": { # Features
        "2017": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.xpt",
    },
    "PAQ": { # Features
        "2017": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_PAQ.xpt",
    },
    "SLQ": { # Features
        "2017": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SLQ.xpt",
    },
    "DBQ": { # Features
        "2017": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DBQ.xpt",
    },
    "BPQ": { # Features
        "2017": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPQ.xpt",
    },
    "SMQ": { # Features
        "2017": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SMQ.xpt",
    }
}


# --- 3. Columnas a Guardar (Features y Labels) ---
# (Esta sección no se modifica)
COLUMNS_TO_SAVE = {
    "BPX": ["SEQN", "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2", "BPXSY3", "BPXDI3", "BPXSY4", "BPXDI4"],
    "PAQ": ["SEQN", "PAQ605", "PAQ610", "PAD615", "PAQ635", "PAQ640", "PAD645", "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670"],
    "SLQ": ["SEQN", "SLD010H", "SLD012", "SLQ050"],
    "DBQ": ['SEQN', 'DBQ010', 'DBD030', 'DBD050', 'DBQ700', 'DBQ197', 'DBQ229', 'DBQ235A', 'DBQ235B', 'DBQ235C', 'DBQ301', 'DBQ330', 'DBQ360', 'DBQ370', 'DBD381', 'DBQ390', 'DBQ400', 'DBD411', 'DBQ421', 'DBQ424', 'DBD895', 'DBD900', 'DBD905', 'DBD910'],
    "BMX": ['SEQN', 'BMDSTATS', 'BMXWT', 'BMIWT', 'BMXRECUM', 'BMIRECUM', 'BMXHEAD', 'BMIHEAD', 'BMXHT', 'BMIHT', 'BMXBMI', 'BMXLEG', 'BMILEG', 'BMXARML', 'BMIARML', 'BMXARMC', 'BMIARMC', 'BMXWAIST', 'BMIWAIST'],
    "DEMO": ['SEQN', 'SDDSRVYR', 'RIDSTATR', 'RIAGENDR', 'RIDAGEYR', 'RIDAGEMN', 'RIDRETH1', 'DMDEDUC2', 'RIDEXPRG', 'SIALANG', 'SIAPROXY', 'SIAINTRP', 'FIALANG', 'FIAPROXY', 'FIAINTRP', 'MIALANG', 'MIAPROXY', 'MIAINTRP', 'SDMVPSU', 'SDMVSTRA', 'INDFMPIR'],
    "BPQ": ['SEQN', 'BPQ020', 'BPQ030', 'BPD035', 'BPQ040A', 'BPQ050A', 'BPQ060', 'BPQ070', 'BPQ080', 'BPQ090D', 'BPQ100D'],
    "SMQ": ['SEQN', 'SMQ020', 'SMD030', 'SMQ040', 'SMQ050Q', 'SMQ050U', 'SMD057', 'SMD641', 'SMD650', 'SMD093', 'SMDUPCA', 'SMD100BR', 'SMD100FL', 'SMD100MN', 'SMD100LN', 'SMD100TR', 'SMD100NI', 'SMD100CO', 'SMD630', 'SMQ670', 'SMAQUEX2']
}