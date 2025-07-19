#!/usr/bin/env python3
"""
Simple event viewer for Orcastrate centralized logging.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_events(events_file="/tmp/orcastrate/events.jsonl"):
    """Load events from JSONL file."""
    events = []
    events_path = Path(events_file)
    
    if not events_path.exists():
        print(f"❌ Events file not found: {events_file}")
        return events
        
    with open(events_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid JSON line: {e}")
                    
    return events


def format_timestamp(iso_timestamp):
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        return dt.strftime('%H:%M:%S')
    except:
        return iso_timestamp


def display_events(events):
    """Display events in a nice format."""
    if not events:
        print("📄 No events found.")
        return
        
    print(f"📊 Found {len(events)} events")
    print("=" * 60)
    
    for event in events:
        event_type = event.get('event_type', 'unknown')
        timestamp = format_timestamp(event.get('timestamp', ''))
        correlation_id = event.get('correlation_id', 'unknown')[:8]
        execution_id = event.get('execution_id', 'unknown')[:8] if event.get('execution_id') else 'N/A'
        
        # Event type emoji
        emoji = {
            'execution_started': '🚀',
            'execution_completed': '✅' if event.get('success') else '❌',
            'step_started': '⚙️',
            'step_completed': '✅' if event.get('success') else '❌'
        }.get(event_type, '📄')
        
        print(f"{emoji} {timestamp} [{correlation_id}] {event_type}")
        
        if event_type == 'execution_started':
            print(f"   📋 Operation: {event.get('operation')}")
            print(f"   📝 Requirements: {event.get('requirements_description')}")
            
        elif event_type == 'execution_completed':
            success = event.get('success', False)
            duration = event.get('duration_seconds', 0)
            artifacts = event.get('artifacts_count', 0)
            print(f"   {'✅' if success else '❌'} Success: {success}")
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   📦 Artifacts: {artifacts}")
            
        elif event_type in ['step_started', 'step_completed']:
            step_name = event.get('step_name', 'Unknown')
            print(f"   🔧 Step: {step_name}")
            if event_type == 'step_completed':
                duration = event.get('duration_seconds', 0)
                print(f"   ⏱️  Duration: {duration:.2f}s")
                
        print()


def main():
    """Main function."""
    events_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/orcastrate/events.jsonl"
    
    print("🔍 Orcastrate Event Viewer")
    print(f"📁 Reading from: {events_file}")
    print()
    
    events = load_events(events_file)
    display_events(events)


if __name__ == "__main__":
    main()