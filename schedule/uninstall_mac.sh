#!/usr/bin/env bash
LABEL="local.nfl-epa.update"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "Removed the daily update job."
