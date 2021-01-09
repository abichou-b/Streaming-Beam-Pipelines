import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import re
from datetime import datetime
import pickle
import nltk
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

PROJECT = 'your_project_id'
TOPIC = 'projects/your_project_id/topics/your_topic'
BUCKET = 'your_bucket'
sw_list = stopwords.words('english')
stemmer = SnowballStemmer('english')


def text_process(element):
    
    """Decode, lowercase, remove non-alphabetic characters and stemming text messages"""
    
    sms_processed = element.decode('utf-8')
    sms_processed = sms_processed.lower()
    sms_processed = re.sub('[^a-z]', ' ', sms_processed)
    sms_processed = ' '.join([stemmer.stem(w) for w in sms_processed.split() if w not in sw_list])
    return {'sms_original':element, 
            'process_time': str(datetime.now()),
            'sms_processed':sms_processed
            }


def import_model(source_blob_name, destination_file_name):
    
    """Downloads a blob from the bucket"""

    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


class LabelPrediction(beam.DoFn):

    def __init__(self):
        self._embed_model = None
        self._class_model = None

    def setup(self):
        
        """Import ML modelS from GCS"""
        
        logging.info("Embedding model initialisation")
        gs_embed_blob = "models/embedding_model.pickle"
        dest_embed_file = gs_embed_blob.split('/')[-1]
        import_model(gs_embed_blob, dest_embed_file)
        self._embed_model = pickle.load(open(dest_embed_file, 'rb'))

        logging.info("Classification model initialisation")
        gs_class_blob = "models/classification_model.pickle"
        dest_class_file = gs_class_blob.split('/')[-1]
        import_model(gs_class_blob, dest_class_file)
        self._class_model = pickle.load(open(dest_class_file, 'rb'))

    def process(self, element):
        
        """Word embedding and label prediction"""
        
        label = self._class_model.predict(self._embed_model.transform([element['sms_processed']]))
        if label==0:
            element['label'] = 'ham'
        else:
            element['label'] = 'spam'
        return [element]


def run(argv=None):
    
    table_schema = 'sms_original:STRING, process_time:STRING, sms_processed:STRING, label:STRING'
    p = beam.Pipeline(options=PipelineOptions())
    (p
     |"Read data from pub/sub input topic" >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes)
     |"Process text data" >> beam.Map(text_process)
     |"Prdict spam/ham label" >> beam.ParDo(LabelPrediction())
     |"Write results to BigQuery" >> beam.io.WriteToBigQuery('{0}:stream_nlp.spamdata'.format(PROJECT),
                                                           schema=table_schema,
                                                           write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
     )
    result = p.run()
    result.wait_until_finish()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()













