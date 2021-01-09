

from google.cloud import pubsub_v1
import random
import time
import pandas as pd


PROJECT_ID ='your_project_id'
TOPIC = 'your_topic'
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC)


if __name__ == '__main__':
    
    df = pd.read_csv('gs://bucket/path/to/sms_unseen.csv')
    while True:
        new_sms = df.sample(n=1)['sms'].to_string(index=False)
        message_future = publisher.publish(topic_path, new_sms.encode('utf-8'))
        print(message_future.result())
        sleep_time = random.choice(range(1, 3))
        time.sleep(sleep_time)