# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
   Author :       Lily Zheng
   date：          2022/8/19 15:29
-------------------------------------------------
   Change Activity:
                   2022/8/19:
-------------------------------------------------
用于MEG Pipeline Report的HTML生成
"""
__author__ = 'Lily Zheng'

import os
import base64
import html
import datetime
import sys

PY_VERSION = sys.version_info

imgpath = './imgs/IED_detection_overview.png'

#
# Tool Functions
#

def convert_img_to_base64(imgpath):
    img_type = os.path.splitext(imgpath)[1][1:]
    with open(imgpath, 'rb') as fid:
        encoded_img = base64.b64encode(fid.read()).decode()
        if PY_VERSION >= (3, 6):
            encoded_img_str = f"data:image/{img_type};base64,{encoded_img}"
        else:
            encoded_img_str = "data:image/;base64,%s"%(img_type,encoded_img)
    return encoded_img_str

def get_file_with_suffix(path, suffix=".css"):
    fileList=os.listdir(path)
    resultList=list()
    for f in fileList:
        if os.path.splitext(f)[1] == suffix:
            resultList.append(os.path.join(path, f))
    return resultList

class Report(object):
    def __init__(self,subject_info,
                 meg_info,
                 coregistration_info,
                 ied_number_info,
                 ied_events_info,
                 ied_clusters_info,
                 preliminary_conclusions_info,
                 analysis_info,
                 reference_info):

        self.html_template = './html_template'
        self.meg_pipeline_results = './meg_pipeline_results'

        self.save_html_path = './meg_report'

        # 报告标题
        self.report_title_info = "MEG Data Analysis Report"

        # 被试基础信息
        self.subject_info = subject_info

        # MEG基础信息
        self.meg_info = meg_info

        # 自动配准信息
        self.coregistration_info = coregistration_info

        # IED Number and Timing
        self.ied_number_info = ied_number_info

        # IED Events Info
        self.ied_events_info = ied_events_info

        # IED Clusters Info
        self.ied_clusters_info = ied_clusters_info

        # Preliminary Conclusions
        self.preliminary_conclusions_info = preliminary_conclusions_info

        # Analysis Information
        self.analysis_info = analysis_info

        # Reference Info
        self.reference_info = reference_info


    def _gen_img_html(self, section_info_dict):
        section_title = section_info_dict["section_title"]

        # Begin Report Item Div
        cHtml='\n<div class="report_item">\n'
        cHtml+="\n"

        ## Report Item Title
        cHtml += '<p class="report_item_title">\n'
        templatePath = self.html_template
        imgPath = os.path.join(templatePath, "source", "section_icon.png")
        imgStr = convert_img_to_base64(imgPath)

        cHtml += '\t<img src="%s" alt="report_item">\n' % imgStr
        cHtml += "\t"+section_title+"\n"
        cHtml += "</p>\n"
        cHtml += "\n"

        for k, v in section_info_dict.items():
            if isinstance(v, dict):
                commentContent = section_info_dict[k]["commentContent"]
                img_path = section_info_dict[k]["img_path"]
                imgStr = convert_img_to_base64(img_path)
                cHtml += '<div class="result_w">\n'
                cHtml += '\t<img class="result_pic" src="%s" alt="%s"/>\n' % (imgStr,  html.escape(commentContent))
                cHtml += '\t<p class="result_caption">%s</p>\n' % (html.escape(commentContent))
                cHtml += "</div>\n"
                cHtml += "\n"

        # End Report Item Div
        cHtml += "</div>\n"

        return cHtml

    def gen_title_html(self):
        """
        生成Html报告的标题代码
        """
        # Begin Title Div
        tHtml='\n<div class="title_w">\n'
        tHtml+="\n"

        ## Title
        tHtml+='\t<p class="title">\n'
        tHtml+="\t\t<span>%s</span>\n" % html.escape(self.report_title_info)
        tHtml+="\t</p>\n"
        tHtml+="\n"

        # End Title Div
        tHtml+="</div>\n"
        tHtml+="\n"

        return tHtml

    def gen_subject_basicinfo_html(self):
        """
        生成被试基础信息的HTML代码
        :return:
        """
        subject_info = self.subject_info
        infoHeader = {}
        infoHeader["subjectID"] = "ID"
        infoHeader["subjectName"] = "Name"
        infoHeader["subjectAge"] = "Age"
        infoHeader["subjectGender"] = "Gender"

        # Begin Report Item Div
        bHtml='\n<div class="report_item">\n'
        bHtml+="\n"

        ## Report Item Title
        bHtml += '<p class="report_item_title">\n'
        templatePath = self.html_template
        imgPath = os.path.join(templatePath, "source", "section_icon.png")
        imgStr = convert_img_to_base64(imgPath)

        bHtml += '\t<img src="%s" alt="report_item">\n' % imgStr
        bHtml += "\t"+"Subject Information"+"\n"
        bHtml += "</p>\n"
        bHtml += "\n"

        # End Report Item Div
        bHtml += "</div>\n"


        # bHtml=''
        bHtml+='<table class="basic_info_table">\n'
        bHtml+="<thead>\n"
        bHtml+="\t<tr>\n"
        for key, header in infoHeader.items():
            bHtml+="\t\t<th>%s</th>\n" % html.escape(header)
        bHtml+="\t</tr>\n"
        bHtml+="</thead>\n"

        bHtml+="<tbody>\n"
        bHtml+="\t<tr>\n"
        for key in infoHeader:
            bHtml+="\t\t<td>%s</td>\n" % html.escape(subject_info[key])
        bHtml+="\t</tr>\n"
        bHtml+="</tbody>\n"
        bHtml+="</table>\n"

        return bHtml

    def gen_meg_basicinfo_html(self):
        """
        生成MEG的基础信息Html代码
        :return:
        """
        meg_info = self.meg_info
        infoHeader = {}
        infoHeader["megSensors"] = "# Sensors"
        infoHeader["megDate"] = "Date"
        infoHeader["megDuration"] = "Duration"
        infoHeader["megSegments"] = "# Segments"

        # Begin Report Item Div
        bHtml='\n<div class="report_item">\n'
        bHtml+="\n"

        ## Report Item Title
        bHtml += '<p class="report_item_title">\n'
        templatePath = self.html_template
        imgPath = os.path.join(templatePath, "source", "section_icon.png")
        imgStr = convert_img_to_base64(imgPath)

        bHtml += '\t<img src="%s" alt="report_item">\n' % imgStr
        bHtml += "\t"+"MEG Information"+"\n"
        bHtml += "</p>\n"
        bHtml += "\n"

        # End Report Item Div
        bHtml += "</div>\n"

        # bHtml=''
        bHtml+='<table class="basic_info_table">\n'
        bHtml+="<thead>\n"
        bHtml+="\t<tr>\n"
        for key, header in infoHeader.items():
            bHtml+="\t\t<th>%s</th>\n" % html.escape(header)
        bHtml+="\t</tr>\n"
        bHtml+="</thead>\n"

        bHtml+="<tbody>\n"
        bHtml+="\t<tr>\n"
        for key in infoHeader:
            bHtml+="\t\t<td>%s</td>\n" % html.escape(meg_info[key])
        bHtml+="\t</tr>\n"
        bHtml+="</tbody>\n"
        bHtml+="</table>\n"

        return bHtml

    def gen_coregistration_html(self):
        return self._gen_img_html(section_info_dict=self.coregistration_info)

    def gen_coregistration_html_(self):
        section_title = self.coregistration_info["section_title"]

        # Begin Report Item Div
        cHtml='\n<div class="report_item">\n'
        cHtml+="\n"

        ## Report Item Title
        cHtml += '<p class="report_item_title">\n'
        templatePath = self.html_template
        imgPath = os.path.join(templatePath, "source", "section_icon.png")
        imgStr = convert_img_to_base64(imgPath)

        cHtml += '\t<img src="%s" alt="report_item">\n' % imgStr
        cHtml += "\t"+section_title+"\n"
        cHtml += "</p>\n"
        cHtml += "\n"

        for k, v in self.coregistration_info.items():
            if isinstance(v, dict):
                commentContent = self.coregistration_info[k]["commentContent"]
                img_path = self.coregistration_info[k]["img_path"]
                imgStr = convert_img_to_base64(img_path)
                cHtml += '<div class="result_w">\n'
                cHtml += '\t<img class="result_pic" src="%s" alt="%s"/>\n' % (imgStr,  html.escape(commentContent))
                cHtml += '\t<p class="result_caption">%s</p>\n' % (html.escape(commentContent))
                cHtml += "</div>\n"
                cHtml += "\n"
        # End Report Item Div
        cHtml += "</div>\n"

        return cHtml

    def gen_ied_number_and_timing_html(self):

        # 棘波概览图
        section_title = self.ied_number_info["section_title"]

        # Begin Report Item Div
        cHtml='\n<div class="report_item ied_number_timing">\n'
        cHtml+="\n"

        ## Report Item Title
        cHtml += '<p class="report_item_title">\n'
        templatePath = self.html_template
        imgPath = os.path.join(templatePath, "source", "section_icon.png")
        imgStr = convert_img_to_base64(imgPath)

        cHtml += '\t<img src="%s" alt="report_item">\n' % imgStr
        cHtml += "\t"+section_title+"\n"
        cHtml += "</p>\n"
        cHtml += "\n"

        for k, v in self.ied_number_info.items():
            if isinstance(v, list) and k == "IED_Timing":
                for imginfo in v:
                    commentContent = imginfo["commentContent"]
                    img_path = imginfo["img_path"]
                    imgStr = convert_img_to_base64(img_path)
                    cHtml += '<div class="result_w">\n'
                    cHtml += '\t<img class="result_pic" src="%s" alt="%s"/>\n' % (imgStr,  html.escape(commentContent))
                    cHtml += '\t<p class="result_caption">%s</p>\n' % (html.escape(commentContent))
                    cHtml += "</div>\n"
                    cHtml += "\n"

        # End Report Item Div
        cHtml += "</div>\n"
        cHtml += "<br>"

        # 统计表格
        itemHeader = {"BR": "Brain regions",
                      "LT": "LT",
                      "RT": "RT",
                      "LP": "LP",
                      "RP": "RP",
                      "LO": "LO",
                      "RO": "RO",
                      "LF": "LF",
                      "RF": "RF"}

        ## Report Item Table Header
        table_header = self.ied_number_info['IED_number']['table_name']

        if len(table_header)!=0 or not table_header.isspace():
            cHtml+='<p class="report_item_header">\n'
            cHtml+='%s\n' % html.escape(table_header)
            cHtml += '<br /><br />'
            cHtml+='</p>'

        ## Report Item Table
        cHtml+='<table class="result_table">\n'
        cHtml+="<thead>\n"
        cHtml+="\t<tr>\n"
        for key, header in itemHeader.items():
            cHtml+="\t\t<th>%s</th>\n" % html.escape(header)
        cHtml+="\t</tr>\n"
        cHtml+="</thead>\n"

        ## Report Item Body
        cHtml+="<tbody>\n"

        itemContent = self.ied_number_info['IED_number']['stats']
        for key in itemContent.keys():
            cHtml+="\t<tr>\n"
            contents = itemContent[key]

            if key == "numbers":
                contents.insert(0,"Number of IED events")
            elif key == "percentage":
                contents.insert(0,"Percentage")

            for content in contents:
                cHtml+="\t\t<td>%s</td>\n" % html.escape(content)
            cHtml+="\t</tr>\n"

        cHtml+="</tbody>\n"
        cHtml+="</table>\n"

        cHtml+='<p class="report_item_header">\n'
        cHtml+='%s\n' % html.escape(self.ied_number_info['IED_number']["comment_content"][0:81])
        cHtml+='</p>'

        cHtml+='<p class="report_item_header">\n'
        cHtml+='%s\n' % html.escape(self.ied_number_info['IED_number']["comment_content"][81:])
        cHtml+='</p>'


        return cHtml

    def gen_source_imaging_ied_events_html(self):
        return self._gen_img_html(section_info_dict=self.ied_events_info)

    def gen_source_imaging_ied_clusters_html(self):
        return self._gen_img_html(section_info_dict=self.ied_clusters_info)

    def gen_preliminary_conclusions_html(self):

        meg_duration_min = self.preliminary_conclusions_info["meg_duration_min"]
        meg_position = self.preliminary_conclusions_info["meg_position"]
        ied_events_num = self.preliminary_conclusions_info["ied_events_num"]
        ied_clusters_num = self.preliminary_conclusions_info["ied_clusters_num"]

        if PY_VERSION >= (3, 6):
            con_meg_info = f"<b>{meg_duration_min}</b> minutes MEG recording suggests a generator at <b>{meg_position}</b>."
            con_ied_info = f"<b>{ied_events_num}</b> IED events and <b>{ied_clusters_num}</b> IED clusters have been found."
        else:
            con_meg_info = "<b>%s</b>  minutes MEG recording suggests a generator at <b>%s</b> ."%(meg_duration_min, meg_position)
            con_ied_info = "<b>%s</b>  IED events and <b>%s</b> IED clusters have been found."%(ied_events_num, ied_clusters_num)

        # Begin Report Item Div
        cHtml='\n<div class="report_item">\n'
        cHtml+="\n"

        ## Report Item Title
        cHtml += '<p class="report_item_title">\n'
        templatePath = self.html_template
        imgPath = os.path.join(templatePath, "source", "section_icon.png")
        imgStr = convert_img_to_base64(imgPath)

        cHtml += '\t<img src="%s" alt="report_item">\n' % imgStr
        cHtml += "\t"+"Preliminary Conclusions"+"\n"
        cHtml += "</p>\n"
        cHtml += "\n"

        cHtml += '\t<p class="reference_caption">%s</p>\n' % (con_meg_info)
        cHtml += '\t<p class="reference_caption">%s</p>\n' % (con_ied_info)

        # End Report Item Div
        cHtml += "</div>\n"

        return cHtml

    def gen_analysis_information_html(self):
        # Begin Report Item Div
        cHtml='\n<div class="report_item">\n'
        cHtml+="\n"

        ## Report Item Title
        cHtml += '<p class="report_item_title">\n'
        templatePath = self.html_template
        imgPath = os.path.join(templatePath, "source", "section_icon.png")
        imgStr = convert_img_to_base64(imgPath)

        cHtml += '\t<img src="%s" alt="report_item">\n' % imgStr
        cHtml += "\t"+"Analysis Information"+"\n"
        cHtml += "</p>\n"
        cHtml += "\n"

        for k,v in self.analysis_info.items():
            cHtml += '\t<p class="reference_caption"><b>%s</b>: %s</p>\n' % (html.escape(k),html.escape(v))

        # End Report Item Div
        cHtml += "</div>\n"

        return cHtml

    def gen_reference_html(self):
        # Begin Report Item Div
        cHtml='\n<div class="report_item">\n'
        cHtml+="\n"

        ## Report Item Title
        cHtml += '<p class="report_item_title">\n'
        templatePath = self.html_template
        imgPath = os.path.join(templatePath, "source", "section_icon.png")
        imgStr = convert_img_to_base64(imgPath)

        cHtml += '\t<img src="%s" alt="report_item">\n' % imgStr
        cHtml += "\t"+"Reference"+"\n"
        cHtml += "</p>\n"
        cHtml += "\n"

        for i in self.reference_info:
            cHtml += '\t<p class="reference_caption">%s</p>\n' % (html.escape(i))

        # End Report Item Div
        cHtml += "</div>\n"
        return cHtml

    def gen_report_head_html(self, loadCss=True, loadJs=True):
        pageTitle=self.report_title_info
        hHtml='\n<meta charset="UTF-8">\n'
        hHtml+='<title>%s</title>\n' % pageTitle

        templatePath=self.html_template
        faviconStr = convert_img_to_base64(os.path.join(templatePath, 'source', 'favicon.png'))
        hHtml+='<link rel="icon" href="%s"/>' % faviconStr

        if loadCss:
            cssFiles=get_file_with_suffix(os.path.join(templatePath, 'css'), '.css')
            for cssPath in cssFiles:
                hHtml+='<style type="text/css">\n'
                hHtml+="\n\t"
                with open(cssPath, "rb") as fid:
                    cssStr=fid.read().decode('utf-8')
                hHtml+=cssStr.replace("\n", "\n\t")
                hHtml+="\n"
                hHtml+='</style>\n'

        if loadJs:
            jsFiles=get_file_with_suffix(os.path.join(templatePath, 'js'), '.js')
            for jsPath in jsFiles:
                hHtml+='<script type="text/javascript">\n'
                hHtml+="\n\t"
                with open(jsPath, "rb") as fid:
                    jsStr=fid.read().decode('utf-8')
                hHtml+=jsStr.replace("\n", "\n\t")
                hHtml+="\n"
                hHtml+='</script>\n'

        return hHtml

    def gen_report_body_html(self):
        bHtml='\n<div class="qc_report_wrap">\n'

        ## Basic Html
        bHtml += self.gen_title_html()
        bHtml += self.gen_subject_basicinfo_html()
        bHtml += self.gen_meg_basicinfo_html()
        bHtml += self.gen_coregistration_html()
        bHtml += self.gen_ied_number_and_timing_html()
        bHtml += self.gen_source_imaging_ied_events_html()
        bHtml += self.gen_source_imaging_ied_clusters_html()
        bHtml += self.gen_preliminary_conclusions_html()
        bHtml += self.gen_analysis_information_html()
        # bHtml += self.gen_reference_html()

        bHtml += "</div>\n"
        return bHtml

    def gen_report_html(self):
        headHtml = self.gen_report_head_html()
        bodyHtml = self.gen_report_body_html()

        pageHtml= \
        """
<!DOCTYPE html>
<html lang="en">

<head>
%s
</head>

<body style="background-color: rgba(0, 226, 193, 0.02);margin: 0;padding: 30px 30px;">
%s
</body>

</html>
        """ % (headHtml.replace("\n", "\n\t"), bodyHtml.replace("\n", "\n\t"))

        return pageHtml


    def save_report_html(self, reportName="report.html"):

        targetPath=self.save_html_path
        if not os.path.exists(targetPath):
            os.mkdir(targetPath)

        ReportHtml=self.gen_report_html()
        with open(os.path.join(targetPath, reportName), "wb") as f:
            f.write(ReportHtml.encode('utf-8'))
        print("[INFO]", datetime.datetime.now(), " Generate MEG Report Successfully.")


if __name__ == "__main__":
    # 报告内容

    meg_pipeline_results = './meg_pipeline_results'  # 存放报告图片的路径
    # 被试基础信息
    subject_info = {"subjectID": "MEG2407",
                             "subjectName": "LQY",
                             "subjectAge": "30",
                             "subjectGender": "Male"}
    # MEG基础信息
    meg_info = {"megSensors": "306",
                         "megDate": "xxxx-xx-xx",
                         "megDuration": "35 minutes",
                         "megSegments": "4"}
    # 自动配准信息
    coregistration_info = {
        "section_title": "MEG-MRI Automated Coregistration",
        "fiducials": {
            "commentContent": "Fig 1-1. Initial labeling of the fiducials",
            "img_path": os.path.join(meg_pipeline_results, "Fig1_1_Initial_labeling_of_the_fiducials.jpg")
        }, "coregistration": {
            "commentContent": "Fig 1-2. Final MEG-MRI coregistration",
            "img_path": os.path.join(meg_pipeline_results, "Fig1_2_Final_MEG_MRI_co_registration.jpg")
        }
    }

    # IED Number and Timing
    ied_number_info = {
        "section_title": "IED Number and Timing",
        "IED_Timing": [{
            "commentContent": "Fig 2-1. IED distribution in various sublobar regions "
                              "over time for segment 1 of MEG recordings",
            "img_path": os.path.join(meg_pipeline_results, "Fig2_1_IED_detection_overview.png")
        },{
            "commentContent": "Fig 2-2. IED distribution in various sublobar regions "
                              "over time for segment 1 of MEG recordings",
            "img_path": os.path.join(meg_pipeline_results, "Fig2_2_IED_detection_overview.png")
        },{
            "commentContent": "Fig 2-3. IED distribution in various sublobar regions "
                              "over time for segment 1 of MEG recordings",
            "img_path": os.path.join(meg_pipeline_results, "Fig2_3_IED_detection_overview.png")
        },{
            "commentContent": "Fig 2-4. IED distribution in various sublobar regions "
                              "over time for segment 1 of MEG recordings",
            "img_path": os.path.join(meg_pipeline_results, "Fig2_4_IED_detection_overview.png")
        }
        ],
        "IED_number": {"stats": {
            "numbers": ["0","200","114","1475","13","89","19","36"], #Order: LT RT	LP	RP	LO	RO	LF	RF
            "percentage": ["0.0%","10.3%","5.9%","75.8%","0.7%","4.6%","0.98%","1.9%"]
        },
            "table_name": "Table 1. Number of epochs tested positive from a particular brain region",
            "comment_content": "LT = left temporal; "
                               "RT = right temporal; "
                               "LP = left parietal; "
                               "RP = right parietal;"
                               "LO = left occipital; "
                               "RO = right occipital; "
                               "LF = left frontal; "
                               "RF = right frontal"
        }
    }

    # IED Events Info
    ied_events_info = {
        "section_title": "Source Imaging of Detected IED Events",
        "ECD": {
            "commentContent": "Fig 3-1. MSI of IED events using ECD",
            "img_path": os.path.join(meg_pipeline_results, "Fig3_1_MSI_of_IED_events_using_ECD.png")
        },
        "dSPM": {
            "commentContent": "Fig 3-2. MSI of IED events using dSPM",
            "img_path": os.path.join(meg_pipeline_results, "Fig3_2_MSI_of_IED_events_using_dSPM.png")
        }
    }

    # IED Clusters Info
    ied_clusters_info = {
        "section_title": "Sensor-level Details of Detected IED Cluster(s)",
        "clusters": {
            "commentContent": "Fig 4-1. Sensor waveforms and topographical maps of the main cluster",
            "img_path": os.path.join(meg_pipeline_results,"Fig4_1_Clustering_analysis_results.png")
        }
    }

    # Preliminary Conclusions
    preliminary_conclusions_info = {
        "meg_duration_min": "35",
        "meg_position": "right parietal",
        "ied_events_num": "201",
        "ied_clusters_num": "1"
    }

    # Analysis Information
    analysis_info = {
        "Name": "Batch MSI analysis.",
        "Description": "Source reconstruction for all the IED events.",
        "Input": "Raw MEG recordings & T1-weighted image.",
        "Output": "MEG-MRI coregistration results; IED detection results; "
                  "MSI of all the detected IED events; Sensor-level results of IED clustering analysis.",
        "Parameters": "Band pass-filtering = 1-100 Hz; Head model = Overlapping sphere; "
                      "Source space = Surface-based cortex; Source imaging algorithms = ECD and dSPM methods."
    }

    # Reference Info
    reference_info = [
        "[1] MEG2224_LQY_EP_1_tsss.fif",
        "[2] MEG2224_LQY_EP_2_tsss.fif",
        "[3] MEG2224_LQY_EP_3_tsss.fif",
        "[4] MEG2224_LQY_EP_4_tsss.fif"
    ]

    report = Report(subject_info=subject_info,
                    meg_info=meg_info,
                    coregistration_info=coregistration_info,
                    ied_number_info=ied_number_info,
                    ied_events_info=ied_events_info,
                    ied_clusters_info=ied_clusters_info,
                    preliminary_conclusions_info=preliminary_conclusions_info,
                    analysis_info=analysis_info,
                    reference_info=reference_info)

    report.save_report_html(reportName="test_report.html")
