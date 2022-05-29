import datetime
import json
import os
import click

@click.command()
@click.option('-f', '--file-name', 'fname', help='virustotal report', required=True)
@click.option('-o', '--output', 'opath', default='.', help='output folder')


def cli(fname, opath):
    hashes = ['md5', 'sha1', 'sha256']
    vt_report = json.load(open(fname, 'r'))['data']['attributes']
    report = {}
    analysis_results = []

    for hash in hashes:
        report[hash] = vt_report[hash]
    report['scan_date'] = str(datetime.datetime.fromtimestamp(vt_report['last_analysis_date']))
    report['first_seen'] = str(datetime.datetime.fromtimestamp(vt_report['first_seen_itw_date']))
    
    for av_vendor in vt_report['last_analysis_results']:
        analysis_results.append([av_vendor, vt_report['last_analysis_results'][av_vendor]])
    report['av_labels'] = analysis_results
    
    if not os.path.exists(opath):
        os.mkdir(opath)

    json.dump(report, open(opath + 'report.json', 'w+'))

if __name__ == '__main__':
    cli()
