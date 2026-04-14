# Records experiment results to CSV and renders ASCII summary tables
import os
import csv
import json


class ResultsReporter:
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.csv_path = os.path.join(output_dir, f"{experiment_name}.csv")
        os.makedirs(output_dir, exist_ok=True)
        self._rows = []

    def log_run(self, config: dict, metrics: dict) -> None:
        row = {}
        row.update(config)
        row.update(metrics)
        self._rows.append(row)
        exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists or os.path.getsize(self.csv_path) == 0:
                writer.writeheader()
            writer.writerow(row)

    def print_table(self) -> None:
        if not self._rows:
            print("No results logged yet.")
            return
        keys = list(self._rows[0].keys())
        col_widths = [max(len(str(k)), max(len(str(r.get(k, ""))) for r in self._rows)) for k in keys]
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header = "|" + "|".join(f" {k:<{w}} " for k, w in zip(keys, col_widths)) + "|"
        print(sep)
        print(header)
        print(sep)
        for row in self._rows:
            line = "|" + "|".join(f" {str(row.get(k, '')):<{w}} " for k, w in zip(keys, col_widths)) + "|"
            print(line)
        print(sep)

    def save_config(self, config: dict) -> None:
        path = os.path.join(self.output_dir, f"{self.experiment_name}_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
