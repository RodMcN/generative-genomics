FROM nvcr.io/nvidia/rapidsai/base:24.10-cuda12.5-py3.11
RUN pip install scanpy shap
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

USER root
WORKDIR /workdir

CMD [ "python", "main.py" ]