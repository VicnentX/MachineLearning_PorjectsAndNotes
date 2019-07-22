# from urllib import request
#
#
# url = "https://www.youtube.com/playlist?list=PL505477785BB204D4"
# with request.urlopen(url) as file:
#     data = file.read()
#     print(data)

import requests
import json
import pprint

with requests.get(url='https://github.com/timeline.json') as response:
    # 读取response里的内容，并转码
    data2 = response.json()
    print(data2)

    print("----")
    pprint.pprint(data2)

    print("---")
    # print json in a nice format
    print(json.dumps(data2, indent=4, sort_keys=True))


