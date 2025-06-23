import streamlit as st
import yaml
from pathlib import Path
from dataclasses import dataclass, field
import threading
import time
import queue

from detector import ShoesAIAnalyzer

@dataclass
class Job:
    id: int
    folder: str
    output: str
    config: dict
    status: str = "queued"
    start_time: float = 0.0
    end_time: float = 0.0
    processed: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)

class JobManager:
    def __init__(self):
        self.jobs: list[Job] = []
        self._queue: queue.Queue[Job] = queue.Queue()
        self._counter = 1
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while True:
            job = self._queue.get()
            if job is None:
                break
            if job.cancel_event.is_set():
                job.status = "canceled"
                continue
            job.status = "running"
            job.start_time = time.time()
            cfg_path = Path(f"config_job_{job.id}.yaml")
            with cfg_path.open("w") as fh:
                yaml.safe_dump(job.config, fh)
            try:
                analyzer = ShoesAIAnalyzer(config_path=str(cfg_path))
                df = analyzer.processFolder(
                    folder=job.folder,
                    output_path=job.output,
                    cancel_event=job.cancel_event,
                )
                job.processed = len(df)
                job.status = "completed" if not job.cancel_event.is_set() else "canceled"
            except Exception as e:
                job.status = "error"
                job.processed = 0
                st.error(str(e))
            finally:
                job.end_time = time.time()
                cfg_path.unlink(missing_ok=True)
            self._queue.task_done()

    def queue_job(self, folder: str, output: str, config: dict) -> Job:
        job = Job(id=self._counter, folder=folder, output=output, config=config)
        self._counter += 1
        self.jobs.append(job)
        self._queue.put(job)
        return job

    def cancel_job(self, job_id: int):
        for job in self.jobs:
            if job.id == job_id:
                job.cancel_event.set()
                if job.status == "queued":
                    job.status = "canceled"
                break

# Initialize manager in session state
if "manager" not in st.session_state:
    st.session_state.manager = JobManager()
manager: JobManager = st.session_state.manager

st.title("ShoesAI Queue")

# Load and edit config
def load_default_config():
    path = Path("config.yaml")
    if path.is_file():
        with path.open() as fh:
            return yaml.safe_load(fh)
    return {}

def config_editor(default_cfg):
    cfg_text = st.text_area("Config YAML", yaml.safe_dump(default_cfg, sort_keys=False), height=400)
    try:
        return yaml.safe_load(cfg_text) or {}
    except yaml.YAMLError as e:
        st.error(f"Erro ao parsear YAML: {e}")
        return default_cfg

cfg = config_editor(load_default_config())

with st.form("queue_form"):
    folder = st.text_input("Pasta com imagens", "images")
    output = st.text_input("Arquivo de saída", "output/processed_dataset.json")
    submitted = st.form_submit_button("Adicionar à fila")
    if submitted:
        manager.queue_job(folder, output, cfg)

st.subheader("Tarefas em Execução")
for job in manager.jobs:
    if job.status == "running":
        dur = time.time() - job.start_time
        if st.button(f"Cancelar {job.id}"):
            manager.cancel_job(job.id)
        st.write(f"ID {job.id} | {job.folder} | {dur:.1f}s")

st.subheader("Fila")
for job in manager.jobs:
    if job.status == "queued":
        if st.button(f"Cancelar {job.id}"):
            manager.cancel_job(job.id)
        st.write(f"ID {job.id} | {job.folder}")

st.subheader("Concluídas")
for job in manager.jobs:
    if job.status in {"completed", "canceled", "error"}:
        dur = job.end_time - job.start_time
        st.write(f"ID {job.id} | {job.status} | {job.processed} imgs | {dur:.1f}s")

# Overall stats
completed = [j for j in manager.jobs if j.status == "completed"]
if completed:
    total_time = sum(j.end_time - j.start_time for j in completed)
    st.write(f"Tempo total de processamento: {total_time:.1f}s para {len(completed)} tarefas")
