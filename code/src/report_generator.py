"""
Modul Report Generator
======================
Menghasilkan laporan deteksi lengkap dalam format JSON dan
visualisasi timeline menggunakan matplotlib.

Sesuai dengan BAB 3.6 Skripsi: Output Sistem
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ReportGenerator:
    """
    Generator laporan hasil proctoring.

    Output:
    - JSON report (detail lengkap)
    - Timeline visualization (PNG)
    - Summary text
    """

    RISK_COLORS_MPL = {
        "normal": "#4CAF50",
        "warning": "#FFC107",
        "suspicious": "#FF9800",
        "high_risk": "#FF9800",
        "critical": "#F44336",
    }

    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(self, results: Dict,
                              filename: str = "session_report.json") -> str:
        """
        Generate JSON report.

        Args:
            results: Output dari ProctorPipeline.get_results()
            filename: Nama file output

        Returns:
            Path ke file report
        """
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        print(f"[Report] JSON report saved: {path}")
        return str(path)

    def generate_timeline(self, results: Dict,
                           filename: str = "timeline.png") -> Optional[str]:
        """
        Generate timeline visualization of cheating scores.

        Args:
            results: Output dari ProctorPipeline.get_results()
            filename: Nama file output

        Returns:
            Path ke file PNG atau None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[Report] matplotlib not available. Skipping timeline.")
            return None

        intervals = results.get("intervals", [])
        if not intervals:
            print("[Report] No interval data for timeline.")
            return None

        # Data
        times = [(iv["start"] + iv["end"]) / 2.0 for iv in intervals]
        scores = [iv["score"] for iv in intervals]
        risks = [iv["risk"] for iv in intervals]
        colors = [self.RISK_COLORS_MPL.get(r, "#999") for r in risks]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle("Automated Proctoring — Session Timeline", fontsize=14, fontweight='bold')

        # ── Score Timeline ──
        ax1 = axes[0]
        ax1.bar(times, scores, width=(times[1] - times[0]) * 0.8 if len(times) > 1 else 5,
                color=colors, alpha=0.8, edgecolor='none')
        ax1.set_ylabel("Cheating Score", fontsize=11)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(min(times) - 5, max(times) + 5)

        # Threshold lines (legacy4 or thesis3 depending on payload)
        thresholds = results.get("scoring", {}).get("thresholds", {}) or {}
        if "warning" in thresholds:
            ax1.axhline(
                y=float(thresholds.get("warning", 0.3)),
                color="#FFC107",
                linestyle="--",
                alpha=0.5,
                label="Warning",
            )
            ax1.axhline(
                y=float(thresholds.get("suspicious", 0.7)),
                color="#F44336",
                linestyle="--",
                alpha=0.5,
                label="Suspicious",
            )
        else:
            ax1.axhline(y=float(thresholds.get("suspicious", 0.3)), color='#FFC107', linestyle='--', alpha=0.5, label='Suspicious')
            ax1.axhline(y=float(thresholds.get("high_risk", 0.6)), color='#FF9800', linestyle='--', alpha=0.5, label='High Risk')
            ax1.axhline(y=float(thresholds.get("critical", 0.8)), color='#F44336', linestyle='--', alpha=0.5, label='Critical')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)

        # ── Risk Level Bar ──
        ax2 = axes[1]
        risk_mapping = {"normal": 0, "warning": 1, "suspicious": 2, "high_risk": 3, "critical": 4}
        risk_values = [risk_mapping.get(r, 0) for r in risks]
        width = (times[1] - times[0]) * 0.95 if len(times) > 1 else 5
        ax2.bar(times, [1] * len(times), width=width,
                color=colors, alpha=0.9, edgecolor='none')
        ax2.set_xlabel("Time (seconds)", fontsize=11)
        ax2.set_ylabel("Risk Level", fontsize=11)
        ax2.set_yticks([])
        ax2.set_xlim(min(times) - 5, max(times) + 5)

        # Legend
        legend_patches = [
            mpatches.Patch(color=c, label=l.replace('_', ' ').title())
            for l, c in self.RISK_COLORS_MPL.items()
        ]
        ax2.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=4)

        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Report] Timeline saved: {path}")
        return str(path)

    def generate_indicator_breakdown(self, results: Dict,
                                      filename: str = "indicators.png") -> Optional[str]:
        """Generate breakdown chart of indicator contributions."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        intervals = results.get("intervals", [])
        if not intervals:
            return None

        # Aggregate indicator scores
        all_indicators = {}
        for iv in intervals:
            for name, value in iv.get("indicators", {}).items():
                if name not in all_indicators:
                    all_indicators[name] = []
                all_indicators[name].append(value)

        if not all_indicators:
            return None

        # Average per indicator
        names = list(all_indicators.keys())
        avg_values = [np.mean(all_indicators[n]) for n in names]
        display_names = [n.replace('_', ' ').title() for n in names]

        # Sort by value
        sorted_pairs = sorted(zip(display_names, avg_values), key=lambda x: x[1], reverse=True)
        display_names, avg_values = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(display_names, avg_values, color='#42A5F5', alpha=0.8)

        # Color high values differently
        for bar, val in zip(bars, avg_values):
            if val > 0.5:
                bar.set_color('#F44336')
            elif val > 0.25:
                bar.set_color('#FF9800')

        ax.set_xlabel("Average Score", fontsize=11)
        ax.set_title("Indicator Contribution Breakdown", fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        path = self.output_dir / filename
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Report] Indicator breakdown saved: {path}")
        return str(path)

    def generate_summary_text(self, results: Dict) -> str:
        """Generate human-readable summary text."""
        overall = results.get("overall_score", {})
        session = results.get("session_info", {})
        events = results.get("events", [])

        lines = [
            "=" * 60,
            "  AUTOMATED PROCTORING SYSTEM — SESSION REPORT",
            "=" * 60,
            f"  Date: {session.get('timestamp', 'N/A')}",
            f"  Source: {session.get('source', 'N/A')}",
            f"  Total Frames: {session.get('total_frames', 0)}",
            f"  FPS: {session.get('fps', 0):.1f}",
            "",
            "  OVERALL ASSESSMENT",
            "-" * 40,
            f"  Score: {overall.get('avg_score', 0):.3f} (avg) / {overall.get('max_score', 0):.3f} (max)",
            f"  Risk Level: {overall.get('overall_risk', 'N/A').upper()}",
            f"  Intervals Analyzed: {overall.get('total_intervals', 0)}",
            "",
        ]

        # Risk distribution
        dist = overall.get("risk_distribution", {})
        if dist:
            lines.append("  RISK DISTRIBUTION")
            lines.append("-" * 40)
            for level, count in dist.items():
                bar = "█" * count
                lines.append(f"  {level:>12}: {count:3d} {bar}")
            lines.append("")

        # Events
        event_types = {}
        for e in events:
            t = e["type"]
            event_types[t] = event_types.get(t, 0) + 1
        if event_types:
            lines.append("  EVENTS DETECTED")
            lines.append("-" * 40)
            for etype, count in sorted(event_types.items(), key=lambda x: -x[1]):
                lines.append(f"  {etype:>20}: {count}")
            lines.append("")

        # Cheating timeline (timestamp ranges by type)
        timeline = (
            results.get("scoring", {}).get("cheating_timeline")
            or results.get("cheating_timeline")
            or []
        )
        if timeline:
            lines.append("  CHEATING TIMELINE")
            lines.append("-" * 40)
            for seg in timeline:
                rng = seg.get("range_mmss", "N/A")
                ctype = seg.get("type", "unknown")
                conf = float(seg.get("avg_type_score", 0.0))
                reasons = seg.get("reasons", [])
                reason_text = ", ".join(reasons) if reasons else "-"
                lines.append(f"  {rng:>13}  {ctype:<24} conf={conf:.2f}")
                lines.append(f"  {'':>13}  reason: {reason_text}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_full_report(self, results: Dict,
                              session_name: Optional[str] = None) -> Dict[str, str]:
        """
        Generate semua komponen report.

        Returns:
            Dictionary path ke setiap file report
        """
        prefix = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        paths = {}

        # JSON
        paths["json"] = self.generate_json_report(results, f"{prefix}_report.json")

        # Timeline
        timeline = self.generate_timeline(results, f"{prefix}_timeline.png")
        if timeline:
            paths["timeline"] = timeline

        # Indicator breakdown
        indicators = self.generate_indicator_breakdown(results, f"{prefix}_indicators.png")
        if indicators:
            paths["indicators"] = indicators

        # Summary text
        summary = self.generate_summary_text(results)
        summary_path = self.output_dir / f"{prefix}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        paths["summary"] = str(summary_path)
        print(f"[Report] Summary saved: {summary_path}")

        print(f"\n[Report] Full report generated with {len(paths)} files.")
        return paths
