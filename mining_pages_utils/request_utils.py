import requests
import os
import json


def getZenonInfo(series):
    if series['pub_key'] == 'ZenonID':
        zenonlink_url = 'https://zenon.dainst.org/api/v1/record?id=' + \
            series['pub_value'] + '&field[]=DAILinks'
        zenonbase_url = 'https://zenon.dainst.org/api/v1/record?id=' + \
            series['pub_value']
        s = requests.Session()
        zenonlinks = s.get(zenonlink_url)
        zenonlinks = json.loads(zenonlinks.text)
        zenonlinks = zenonlinks['records'][0]
        zenonbase = s.get(zenonbase_url)
        zenonbase = json.loads(zenonbase.text)
        zenonbase = zenonbase['records'][0]
        zenonbase['gazetteerlinks'] = zenonlinks['DAILinks']['gazetteer']
        zenonbase['thesaurilinks'] = zenonlinks['DAILinks']['thesauri']
        series['pub_info'] = zenonbase
        print('For pub_key: ' + series['pub_key'] + ' found pub_info!')
    else:
        series['pub_info'] = {}
        print('For pub_key: ' + series['pub_key'] + ' no pub_info')
    return series
