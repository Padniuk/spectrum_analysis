Create and set env:
```bash
# For Linux
python3 -m venv env
source env/bin/activate
```
```powershell
# For PowerShell
python -m venv env
.\env\Scripts\Activate
```

Install necessary modules:
```bash
pip install -r requirements.txt 
```

Do not forget to create `.env` file and fill the ploting variables and derivative slope strength

Run script with help flag:
```bash
# For Linux
python3 main.py --help
```
```powershell
# For PowerShell
python main.py --help