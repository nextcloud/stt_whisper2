FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN \
  apt update && \
  apt install -y python3 python3-pip

# Download and install FRP client into /usr/local/bin.
RUN set -ex; \
    ARCH=$(uname -m); \
    if [ "$ARCH" = "aarch64" ]; then \
      FRP_URL="https://raw.githubusercontent.com/nextcloud/HaRP/main/exapps_dev/frp_0.61.1_linux_arm64.tar.gz"; \
    else \
      FRP_URL="https://raw.githubusercontent.com/nextcloud/HaRP/main/exapps_dev/frp_0.61.1_linux_amd64.tar.gz"; \
    fi; \
    echo "Downloading FRP client from $FRP_URL"; \
    curl -L "$FRP_URL" -o /tmp/frp.tar.gz; \
    tar -C /tmp -xzf /tmp/frp.tar.gz; \
    mv /tmp/frp_0.61.1_linux_* /tmp/frp; \
    cp /tmp/frp/frpc /usr/local/bin/frpc; \
    chmod +x /usr/local/bin/frpc; \
    rm -rf /tmp/frp /tmp/frp.tar.gz

COPY requirements.txt /
COPY healthcheck.sh /
COPY --chmod=775 start.sh /

ADD cs[s] /app/css
ADD im[g] /app/img
ADD j[s] /app/js
ADD l10[n] /app/l10n
ADD li[b] /app/lib
ADD model[s] /app/models

RUN \
  python3 -m pip install -r requirements.txt && rm -rf ~/.cache && rm requirements.txt

WORKDIR /app/lib
ENTRYPOINT ["/start.sh", "python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/stt_whisper2
HEALTHCHECK --interval=2s --timeout=2s --retries=300 CMD /healthcheck.sh
