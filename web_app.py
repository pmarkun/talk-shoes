import streamlit as st
import yaml
from pathlib import Path
from dataclasses import dataclass, field
import threading
import time
import queue
from streamlit_file_browser import st_file_browser

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
    total: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobManager:
    def __init__(self):
        self.jobs: list[Job] = []
        self._queue: queue.Queue[Job] = queue.Queue()
        self._counter = 1
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _count_images(self, folder: str) -> int:
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        try:
            return sum(
                1 for p in Path(folder).rglob("*") if p.suffix.lower() in valid_ext
            )
        except Exception:
            return 0

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

                def update(p, t):
                    job.processed = p
                    job.total = t or job.total

                df = analyzer.processFolder(
                    folder=job.folder,
                    output_path=job.output,
                    cancel_event=job.cancel_event,
                    progress_callback=update,
                )
                job.processed = len(df)
                job.status = (
                    "completed" if not job.cancel_event.is_set() else "canceled"
                )
            except Exception as e:
                job.status = "error"
                job.processed = 0
                st.error(str(e))
            finally:
                job.end_time = time.time()
                cfg_path.unlink(missing_ok=True)
            self._queue.task_done()

    def queue_job(self, folder: str, output: str, config: dict) -> Job:
        total = self._count_images(folder)
        job = Job(
            id=self._counter, folder=folder, output=output, config=config, total=total
        )
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


def _edit_dict(d: dict, container=st.sidebar, prefix=""):
    for k, v in d.items():
        key_name = f"{prefix}{k}"
        if isinstance(v, dict):
            with container.expander(k):
                d[k] = _edit_dict(v, container, prefix=f"{key_name}.")
        elif isinstance(v, bool):
            d[k] = container.checkbox(k, value=v, key=key_name)
        elif isinstance(v, int):
            d[k] = int(container.number_input(k, value=v, step=1, key=key_name))
        elif isinstance(v, float):
            d[k] = float(container.number_input(k, value=v, key=key_name))
        elif isinstance(v, list):
            text = container.text_input(k, ",".join(map(str, v)), key=key_name)
            d[k] = [item.strip() for item in text.split(",") if item.strip()]
        else:
            d[k] = container.text_input(k, str(v), key=key_name)
    return d


def config_editor(default_cfg):
    return _edit_dict(default_cfg.copy())


cfg = config_editor(load_default_config())

with st.form("queue_form"):
    st.write("Selecionar pasta de imagens")
    event = st_file_browser(
        ".", show_choose_folder=True, show_choose_file=False, key="folder_browser"
    )
    folder = (
        event.get("target", {}).get("path")
        if event and event.get("type") == "SELECT_FOLDER"
        else ""
    )
    st.write("Selecionar pasta de saída")
    out_event = st_file_browser(
        ".", show_choose_folder=True, show_choose_file=False, key="output_browser"
    )
    output = (
        out_event.get("target", {}).get("path")
        if out_event and out_event.get("type") == "SELECT_FOLDER"
        else ""
    )
    submitted = st.form_submit_button("Adicionar à fila")
    if submitted and folder and output:
        manager.queue_job(folder, output, cfg)

st.subheader("Config YAML")
st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")

st.subheader("Tarefas em Execução")
for job in manager.jobs:
    if job.status == "running":
        dur = time.time() - job.start_time
        if st.button(f"Cancelar {job.id}"):
            manager.cancel_job(job.id)
        st.write(f"ID {job.id} | {job.folder} | {dur:.1f}s")
        progress = job.processed / job.total if job.total else 0.0
        st.progress(progress, text=f"{job.processed}/{job.total}")

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
        if job.total:
            st.progress(1.0, text=f"{job.processed}/{job.total}")

# Overall stats
completed = [j for j in manager.jobs if j.status == "completed"]
if completed:
    total_time = sum(j.end_time - j.start_time for j in completed)
    st.write(
        f"Tempo total de processamento: {total_time:.1f}s para {len(completed)} tarefas"
    )
