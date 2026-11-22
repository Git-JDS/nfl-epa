#!/usr/bin/env bash
# Install a macOS launchd job that runs ../update.sh twice a week at 7:00 AM local time:
#   Tuesday  - after Monday Night Football, so the full Thu/Sun/Mon week is in
#   Friday   - picks up Thursday Night Football
# nflverse refreshes play-by-play overnight after games. If the Mac is asleep at 7:00,
# launchd runs the job when it wakes. Logs go to ~/Library/Logs/nfl-epa-update.log (outside iCloud, which launchd cannot write to).
# Re-run after moving this folder. Remove with ./uninstall_mac.sh.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="local.nfl-epa.update"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
HOUR="${1:-7}"
mkdir -p "$HOME/Library/Logs" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DIR/update.sh</string></array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/nfl-epa-update.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/nfl-epa-update.log</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed: $DIR/update.sh runs Tuesdays and Fridays at ${HOUR}:00. Log: ~/Library/Logs/nfl-epa-update.log"
