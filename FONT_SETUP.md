# Font Configuration for Cross-Platform PDF Generation

The project uses a shared font stack for consistent PDF rendering across Windows and Ubuntu:
- **Segoe UI** (English/Latin)
- **Noto Sans CJK KR** / **Noto Sans KR** (Korean)

## Setup Instructions

### Windows
Segoe UI is pre-installed. Run:
```bash
python scripts/build_report.py cM_1
```

### Ubuntu
Install required fonts first:
```bash
sudo apt update
sudo apt install ttf-mscorefonts-installer fonts-noto-cjk
sudo fc-cache -fv
```

Then generate reports:
```bash
python scripts/build_report.py cM_1
```

## How It Works

The `font-config.css` file is automatically applied during PDF generation via `md-to-pdf`. Both Windows and Ubuntu will use the same font stack, ensuring consistent output across platforms.
