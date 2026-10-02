"""
GEE Authentication helper.
Forces the browser open via PowerShell so the localhost OAuth flow works on Windows.
"""
import subprocess
import webbrowser

# Patch webbrowser so it uses PowerShell Start-Process (reliable on Windows)
_orig_open = webbrowser.open
def _win_open(url, new=0, autoraise=True):
    try:
        subprocess.Popen(
            ["powershell", "-command", f'Start-Process "{url}"'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return _orig_open(url, new, autoraise)

webbrowser.open = _win_open

import ee

print("=" * 60)
print("  LandSentry — Google Earth Engine Authentication")
print("=" * 60)
print()
print("A browser window will open. Please:")
print("  1. Sign in with your Google account")
print("  2. Click 'Allow' on the permissions screen")
print("  3. Wait — authentication completes automatically!")
print()

try:
    ee.Authenticate(auth_mode="localhost", force=True)
    print()
    print("SUCCESS! Credentials saved.")
    print("Testing connection...")
    ee.Initialize(project="landsentry-508714")
    print("Earth Engine initialized successfully!")
    print("You can now run the dashboard.")
except Exception as e:
    print(f"\nERROR: {e}")
    print("\nPlease make sure:")
    print("  - Earth Engine API is enabled at:")
    print("    https://console.cloud.google.com/apis/library/earthengine.googleapis.com?project=landsentry-508714")
    print("  - You are signed in with the correct Google account")
