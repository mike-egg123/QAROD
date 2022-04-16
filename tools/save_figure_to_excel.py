# import pandas as pd
# df = pd.DataFrame({'Year': [2000, 2001, 2002 , 2003]})
# df['link'] = '-'
# df.at[0, 'link'] = '=HYPERLINK("../data/qar/148965-20160126_anomaly_detection_figs/ALT_QNH.jpg", 2000)'
# df.at[1, 'link'] = '=HYPERLINK("https://en.wikipedia.org/wiki/2001", 2001)'
# df.at[2, 'link'] = '=HYPERLINK("https://en.wikipedia.org/wiki/2002", 2002)'
# df.at[3, 'link'] = '=HYPERLINK("https://en.wikipedia.org/wiki/2003", 2003)'
# df.to_excel('test.xlsx', index = False)

import openpyxl
from PIL.Image import Image
from openpyxl import load_workbook
from openpyxl import Workbook

wb = load_workbook('../data/qar/148965-20160126_qar_report.xlsx')
ws = wb.active

img=openpyxl.drawing.image.Image('../data/qar/148965-20160126_anomaly_detection_figs/N11.jpg')
ws.add_image(img, "A5")
wb.save("test1.xlsx")



