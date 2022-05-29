import datetime
import json
import os
import click
from pathlib import Path

def convert_vt_report(fname):
    vt_report = json.load(open(fname, 'r'))['data']['attributes']
    report = {}
    analysis_results = []
    hashes = ['sha1', 'sha256', 'md5']
    
    for hash in hashes:
        report[hash] = vt_report[hash]
    report['scan_date'] = str(datetime.datetime.fromtimestamp(vt_report['last_analysis_date']))
    report['first_seen'] = str(datetime.datetime.fromtimestamp(vt_report.get('first_seen_itw_date', vt_report['first_submission_date'])))
    
    for av_vendor in vt_report['last_analysis_results']:
        analysis_results.append([av_vendor, vt_report['last_analysis_results'][av_vendor]['result']])
    report['av_labels'] = analysis_results
    return report

@click.command()
@click.option('-i', '--input', 'ipath', help='input directory / file', required=True)
@click.option('-o', '--output', 'opath', default='.', help='output folder')

def cli(ipath, opath):
    ipath = Path(ipath)
    opath = Path(opath)
    
    if not os.path.exists(opath):
        os.mkdir(opath)
    
    #directory
    if os.path.isdir(ipath):
        #with open(opath / 'reports.json', 'a+') as o:
        reports = ''
        for f in os.listdir(ipath):
            if not os.path.islink(ipath / f) and not os.path.isdir(ipath / f):
                reports += '\n' + json.dumps(convert_vt_report(ipath / f))
        f = open(opath / 'reports.json', 'w+')
        f.write(reports)
    # file
    elif os.path.exists(ipath):
        json.dump(convert_vt_report(ipath / f), open(opath / 'report.json', 'w+'))
    else:
        print(f'[Error] No such file or directory: {ipath}')

if __name__ == '__main__':
    cli()
