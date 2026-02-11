# The Switchboard - Unified LLM API Gateway
# Multi-stage build for minimal image size

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir hatch

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build wheel
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels .

# Production stage
FROM python:3.11-slim as production

LABEL org.opencontainers.image.title="The Switchboard"
LABEL org.opencontainers.image.description="Unified LLM API Gateway with fail-closed semantics"
LABEL org.opencontainers.image.version="0.1.0"

# Create non-root user
RUN groupadd -r switchboard && useradd -r -g switchboard switchboard

WORKDIR /app

# Install runtime dependencies
COPY --from=builder /app/wheels /app/wheels
RUN pip install --no-cache-dir /app/wheels/*.whl && rm -rf /app/wheels

# Switch to non-root user
USER switchboard

# Default environment (fail-closed in production)
ENV SWITCHBOARD_HOST=0.0.0.0
ENV SWITCHBOARD_PORT=8080
ENV APERION_ALLOW_ECHO=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

EXPOSE 8080

# Run the service
CMD ["python", "-m", "aperion_switchboard.main"]
