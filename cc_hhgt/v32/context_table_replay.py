"""Bounded comparison of typed public/context aggregates against source replay."""
from pathlib import Path
import pyarrow.parquet as pq
from .drug_sparse_query import _bounded_duckdb, _parquet_file_relation


def relation(path):return _parquet_file_relation([Path(path)])


def compare_tables(left,right,keys,*,label="Context"):
    a=pq.ParquetFile(left).schema_arrow;b=pq.ParquetFile(right).schema_arrow
    if not a.equals(b,check_metadata=False) or not set(keys).issubset(a.names):raise ValueError(label+" replay schema differs")
    with _bounded_duckdb() as con:
        for name,path in (("a",left),("b",right)):
            con.execute(f"CREATE VIEW {name} AS SELECT *,true AS _present FROM {relation(path)}")
            if con.execute(f"SELECT count(*) FROM (SELECT {','.join(keys)} FROM {name} GROUP BY ALL HAVING count(*)<>1)").fetchone()[0]:
                raise ValueError(label+" replay has duplicate keys")
        equal=[]
        for field in a:
            col='"'+field.name.replace('"','""')+'"';expr=f"a.{col} IS NOT DISTINCT FROM b.{col}"
            if str(field.type) in {"double","float"}:
                expr=f"({expr}) OR (isfinite(a.{col}) AND isfinite(b.{col}) AND abs(a.{col}-b.{col})<=1e-12)"
            equal.append(f"coalesce(({expr}),false)")
        join=' AND '.join(f'a.{key} IS NOT DISTINCT FROM b.{key}' for key in keys)
        if con.execute(f"SELECT count(*) FROM a FULL JOIN b ON {join} WHERE a._present IS NULL OR b._present IS NULL OR NOT ({' AND '.join(equal)})").fetchone()[0]:
            raise ValueError(label+" replay differs from published values")
