# Lambda image for background removal (handler.py). Build from the repo root:
#   docker build -f lambda/rembg/Dockerfile -t lovablepet-rembg .
# Test locally with the runtime emulator in the base image:
#   docker run -p 9000:8080 -e REMBG_KEY=test lovablepet-rembg
#   REMBG_URL=http://localhost:9000/2015-03-31/functions/function/invocations REMBG_KEY=test ...
FROM public.ecr.aws/lambda/python:3.13

RUN pip install --no-cache-dir onnxruntime==1.30.0 numpy==2.4.4 pillow==12.3.0

# the model rembg downloads for bria-rmbg, checked against rembg's own sha256
RUN python -c "import hashlib, urllib.request; \
p = '/opt/bria-rmbg.onnx'; \
urllib.request.urlretrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg-2.0.onnx', p); \
h = hashlib.sha256(open(p, 'rb').read()).hexdigest(); \
assert h == '5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958', h" \
    && chmod 644 /opt/bria-rmbg.onnx

COPY api/bria.py lambda/rembg/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
