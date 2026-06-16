#!/bin/bash

# Ensure the script is run with root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Please execute it with sudo."
   exit 1
fi

case "$1" in
    --enable)
        echo "Enabling high performance mode..."
        
        # Set power profile to performance
        if command -v powerprofilesctl &> /dev/null; then
            powerprofilesctl set performance
            echo "Power profile successfully set to 'performance'."
        else
            echo "Warning: powerprofilesctl not found. Is power-profiles-daemon installed?"
        fi
        
        # Disable sleep, suspend, and hibernation via systemd
        systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
        echo "Sleep, suspend, and hibernation targets have been masked (disabled)."
        ;;
        
    --disable)
        echo "Restoring laptop to normal condition..."
        
        # Restore power profile to balanced
        if command -v powerprofilesctl &> /dev/null; then
            powerprofilesctl set balanced
            echo "Power profile successfully restored to 'balanced'."
        fi
        
        # Re-enable sleep, suspend, and hibernation
        systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
        echo "Sleep, suspend, and hibernation targets have been unmasked (enabled)."
        ;;
        
    *)
        echo "Usage: $0 [--enable | --disable]"
        exit 1
        ;;
esac