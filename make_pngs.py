import subprocess, os, fitz

docx = r"C:\Users\PRABHAKAR\Documents\Wells-Lab\PROJECT_REPORT.docx"
pdf  = r"C:\Users\PRABHAKAR\Documents\Wells-Lab\PROJECT_REPORT.pdf"
pdir = r"C:\Users\PRABHAKAR\Documents\Wells-Lab\report_pages"
os.makedirs(pdir, exist_ok=True)

# Step 1 – DOCX -> PDF via Word COM
ps = (
    '$w=New-Object -ComObject Word.Application;$w.Visible=$false;'
    '$d=$w.Documents.Open(\'' + docx.replace("\\","\\\\") + '\');'
    '$d.ExportAsFixedFormat(\'' + pdf.replace("\\","\\\\") + '\',17);'
    '$d.Close();$w.Quit();'
    '[System.Runtime.Interopservices.Marshal]::ReleaseComObject($w)|Out-Null;'
    'Write-Output "PDF_DONE"'
)
r = subprocess.run(["powershell","-Command",ps], capture_output=True, text=True, timeout=90)
print("PDF result:", r.stdout.strip())
if "PDF_DONE" not in r.stdout:
    print("ERR:", r.stderr[:400])
    raise SystemExit(1)

# Step 2 – PDF -> PNG
doc = fitz.open(pdf)
print(f"Pages: {len(doc)}")
for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    out = os.path.join(pdir, f"page_{i+1:02d}.png")
    pix.save(out)
    print(f"  page {i+1:02d} saved  ({os.path.getsize(out)//1024} KB)")
doc.close()
print("DONE")
