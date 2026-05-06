"""
Export Module
Generate various export formats from segmentation results.
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from .segmentation import SegmentationResult, VideoSegment, SegmentType
except ImportError:
    from segmentation import SegmentationResult, VideoSegment, SegmentType


SEGMENT_LABELS = {
    SegmentType.CORE_CONTENT: "Content",
    SegmentType.AD: "Advertisement",
    SegmentType.INTRO: "Intro",
    SegmentType.OUTRO: "Outro",
    SegmentType.TRANSITION: "Transition",
    SegmentType.SILENCE: "Silence",
    SegmentType.RECAP: "Recap",
    SegmentType.UNKNOWN: "Unknown"
}


def format_time_youtube(seconds: float) -> str:
    """Format time for YouTube chapters (M:SS or H:MM:SS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_time_srt(seconds: float) -> str:
    """Format time for SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def export_youtube_chapters(result: SegmentationResult, 
                            include_ads: bool = False) -> str:
    """
    Generate YouTube chapter markers.
    
    Args:
        result: Segmentation result
        include_ads: Whether to include ad segments in chapters
        
    Returns:
        String with YouTube chapter format
    """
    lines = ["Chapters:"]
    lines.append(f"0:00 Start")
    
    for seg in result.segments:
        if seg.segment_type == SegmentType.AD and not include_ads:
            continue
        
        time_str = format_time_youtube(seg.start_time)
        label = SEGMENT_LABELS.get(seg.segment_type, "Segment")
        
        if seg.segment_type == SegmentType.CORE_CONTENT:
            label = f"Content Part {seg.features.get('segment_index', '')}"
        
        lines.append(f"{time_str} {label}")
    
    return "\n".join(lines)


def export_srt_chapters(result: SegmentationResult, output_path: str):
    """
    Export chapters as SRT subtitle file.
    
    Args:
        result: Segmentation result
        output_path: Path to save SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(result.segments, 1):
            start = format_time_srt(seg.start_time)
            end = format_time_srt(seg.end_time)
            label = SEGMENT_LABELS.get(seg.segment_type, "Segment")
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"[{label}]\n\n")


def export_json_metadata(result: SegmentationResult, output_path: str):
    """Export full metadata as JSON."""
    data = result.to_dict()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def export_edl(result: SegmentationResult, output_path: str, 
               fps: float = 30.0):
    """
    Export as EDL (Edit Decision List) for video editors.
    
    Args:
        result: Segmentation result
        output_path: Path to save EDL file
        fps: Video frame rate
    """
    def timecode(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * fps)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
    
    with open(output_path, 'w') as f:
        f.write("TITLE: Video Segmentation\n")
        f.write(f"FCM: NON-DROP FRAME\n\n")
        
        for i, seg in enumerate(result.segments, 1):
            label = SEGMENT_LABELS.get(seg.segment_type, "SEGMENT")
            f.write(f"{i:03d}  AX       V     C        ")
            f.write(f"{timecode(seg.start_time)} {timecode(seg.end_time)} ")
            f.write(f"{timecode(seg.start_time)} {timecode(seg.end_time)}\n")
            f.write(f"* {label}\n\n")


def export_csv(result: SegmentationResult, output_path: str):
    """Export segments as CSV."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("segment_num,type,start_time,end_time,duration,start_formatted,end_formatted\n")
        
        for i, seg in enumerate(result.segments, 1):
            label = SEGMENT_LABELS.get(seg.segment_type, "Unknown")
            start_fmt = format_time_youtube(seg.start_time)
            end_fmt = format_time_youtube(seg.end_time)
            
            f.write(f"{i},{label},{seg.start_time:.3f},{seg.end_time:.3f},")
            f.write(f"{seg.duration:.3f},{start_fmt},{end_fmt}\n")


def export_html_report(result: SegmentationResult, output_path: str):
    """Generate HTML report with visual timeline."""
    
    summary = result.to_dict()['summary']
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Video Segmentation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1e1e1e; color: white; }}
        h1 {{ color: #4CAF50; }}
        .stats {{ background: #2d2d2d; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .timeline {{ height: 50px; background: #333; border-radius: 4px; margin: 20px 0; display: flex; }}
        .segment {{ height: 100%; }}
        .content {{ background: #4CAF50; }}
        .ad {{ background: #f44336; }}
        .intro {{ background: #2196F3; }}
        .outro {{ background: #9C27B0; }}
        .transition {{ background: #FFC107; }}
        .silence {{ background: #9E9E9E; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
        th {{ background: #333; }}
        .legend {{ display: flex; gap: 20px; margin: 20px 0; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Video Segmentation Report</h1>
    
    <div class="stats">
        <h3>Summary</h3>
        <p><strong>Duration:</strong> {format_time_youtube(result.duration)}</p>
        <p><strong>Total Segments:</strong> {len(result.segments)}</p>
        <p><strong>Content Ratio:</strong> {summary['content_ratio']:.1%}</p>
        <p><strong>Ad Ratio:</strong> {summary['ad_ratio']:.1%}</p>
    </div>
    
    <h3>Visual Timeline</h3>
    <div class="legend">
        <div class="legend-item"><div class="legend-color content"></div> Content</div>
        <div class="legend-item"><div class="legend-color ad"></div> Advertisement</div>
        <div class="legend-item"><div class="legend-color intro"></div> Intro</div>
        <div class="legend-item"><div class="legend-color outro"></div> Outro</div>
    </div>
    
    <div class="timeline">
"""
    
    for seg in result.segments:
        width = (seg.duration / result.duration) * 100
        css_class = seg.segment_type.value.replace('_', '-')
        if css_class == "core-content":
            css_class = "content"
        html += f'        <div class="segment {css_class}" style="width: {width}%;" title="{SEGMENT_LABELS[seg.segment_type]}: {format_time_youtube(seg.start_time)} - {format_time_youtube(seg.end_time)}"></div>\n'
    
    html += """    </div>
    
    <h3>Segment Details</h3>
    <table>
        <tr>
            <th>#</th>
            <th>Type</th>
            <th>Start</th>
            <th>End</th>
            <th>Duration</th>
        </tr>
"""
    
    for i, seg in enumerate(result.segments, 1):
        label = SEGMENT_LABELS.get(seg.segment_type, "Unknown")
        html += f"""        <tr>
            <td>{i}</td>
            <td>{label}</td>
            <td>{format_time_youtube(seg.start_time)}</td>
            <td>{format_time_youtube(seg.end_time)}</td>
            <td>{seg.duration:.1f}s</td>
        </tr>
"""
    
    html += """    </table>
    
    <h3>YouTube Chapters</h3>
    <pre style="background: #333; padding: 15px; border-radius: 8px;">"""
    
    html += export_youtube_chapters(result)
    
    html += """</pre>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def export_all(result: SegmentationResult, output_dir: str, base_name: str):
    """Export all formats to a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    export_json_metadata(result, os.path.join(output_dir, f"{base_name}.json"))
    export_csv(result, os.path.join(output_dir, f"{base_name}.csv"))
    export_srt_chapters(result, os.path.join(output_dir, f"{base_name}_chapters.srt"))
    export_html_report(result, os.path.join(output_dir, f"{base_name}_report.html"))
    
    chapters_path = os.path.join(output_dir, f"{base_name}_youtube_chapters.txt")
    with open(chapters_path, 'w') as f:
        f.write(export_youtube_chapters(result))
    
    print(f"Exported to {output_dir}:")
    print(f"  - {base_name}.json (full metadata)")
    print(f"  - {base_name}.csv (spreadsheet)")
    print(f"  - {base_name}_chapters.srt (subtitles)")
    print(f"  - {base_name}_report.html (visual report)")
    print(f"  - {base_name}_youtube_chapters.txt (YouTube)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = SegmentationResult.load(sys.argv[1])
        base_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
        export_all(result, "exports", base_name)
