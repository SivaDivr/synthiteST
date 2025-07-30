
$folderPath = "C:\Datadivr\synthiteST-main"
Set-Location $folderPath


& "$folderPath\stenv\Scripts\Activate.ps1"

streamlit run appst.py
