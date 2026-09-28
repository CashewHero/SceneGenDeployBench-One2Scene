FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel AS curope-builder

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/model/encoder/backbone/croco/curope /build/curope
RUN python -m pip install --no-cache-dir build ninja \
    && python -m pip wheel --no-deps --wheel-dir /wheels /build/curope


FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

WORKDIR /app/model_repo

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY runner_wrapper/requirements.txt /tmp/runner-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/runner-requirements.txt

COPY --from=curope-builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

COPY . /app/model_repo

ENV PYTHONPATH=/app/model_repo:/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUNNER_PORT=58090
ENV RUNNER_NAME=one2scene-scaffold
ENV RUNNER_TYPE=generator
ENV RUNNER_VERSION=0.1.0
ENV RUNNER_CONTRACT_VERSION=1
ENV RUNNER_IDLE_TIMEOUT_SECONDS=900
ENV RUNNER_ADAPTER=runner_wrapper.adapter:run_job
ENV HF_HUB_DISABLE_XET=1

EXPOSE 58090

CMD ["python", "-m", "runner_wrapper.server"]
