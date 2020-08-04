from __future__ import (absolute_import, division, print_function, unicode_literals)

import urllib3

url1 = 'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531809/round1/amap_traffic_annotations_train.json'
url2 = 'https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531809/round1/amap_traffic_annotations_test.json'
http = urllib3.PoolManager()
response = http.request('GET', url1)
with open('./data/amap_traffic_annotations_train.json', 'wb') as f:
    f.write(response.data)
response = http.request('GET', url2)
with open('./data/amap_traffic_annotations_test.json', 'wb') as f:
    f.write(response.data)
response.release_conn()