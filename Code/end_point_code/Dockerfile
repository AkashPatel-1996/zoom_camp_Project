FROM python:3.10

RUN mkdir -p /app/Code/Flask_Code


COPY ./predict.py /app/Code/Flask_Code

COPY templates /app/Code/Flask_Code/templates

COPY ./requirments.txt /app/Code/Flask_Code/requirments.txt

WORKDIR /app/Code/Flask_Code

RUN pip install --no-cache-dir --upgrade -r /app/Code/Flask_Code/requirments.txt



EXPOSE 9691

COPY lin_beg.bin /app/SavedModel/lin_beg.bin

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9691", "predict:app" ]