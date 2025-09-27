import asyncio

import pandas as pd
from fastapi import FastAPI, Form, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from stream_tools import (
    StreamSimulator,
    TimeSeriesBuffer,
    check_alerts,
    compute_metrics_on_window,
    resample_uniform,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILES = {"fhr": None, "uc": None}


@app.post("/upload")
async def upload_files(fhr_file: UploadFile, uc_file: UploadFile):

    df_fhr = pd.read_csv(fhr_file.file)
    df_uc = pd.read_csv(uc_file.file)
    for df in (df_fhr, df_uc):
        df.columns = df.columns.str.strip()
    DATA_FILES["fhr"] = df_fhr[["time_sec", "value"]].dropna().reset_index(drop=True)
    DATA_FILES["uc"] = df_uc[["time_sec", "value"]].dropna().reset_index(drop=True)
    return {"rows_fhr": len(DATA_FILES["fhr"]), "rows_uc": len(DATA_FILES["uc"])}


@app.websocket("/ws/stream")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    sim_fhr = StreamSimulator(DATA_FILES["fhr"], step_sec=1.0, agg="mean")
    sim_uc = StreamSimulator(DATA_FILES["uc"], step_sec=1.0, agg="mean")
    buf_fhr = TimeSeriesBuffer(120)

    while not sim_fhr.finished() and not sim_uc.finished():
        fhr_agg, fhr_raw = sim_fhr.step_one_second()
        uc_agg, uc_raw = sim_uc.step_one_second()

        # для метрик — используем все сырые точки
        for t, v in fhr_raw:
            buf_fhr.append(t, v)

        # метрики
        metrics, alerts = None, []
        t_win, v_win = buf_fhr.as_arrays()
        if len(t_win) > 8:
            tu, vu = resample_uniform(t_win, v_win, fs=4.0, gap_sec=3.0)
            metrics = compute_metrics_on_window(tu, vu)
            # if metrics:
            #     alerts = check_alerts(metrics)

        await ws.send_json(
            {
                "fhr_points": [fhr_agg] if fhr_agg else [],  # 1 точка для графика
                "uc_points": [uc_agg] if uc_agg else [],
                "metrics": metrics.__dict__ if metrics else None,
                "alerts": alerts,
            }
        )

        await asyncio.sleep(1.0)
