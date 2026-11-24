import argparse
from chains.ingestion_chain import ingest_sources
from main import build_runtime
def main():
    p=argparse.ArgumentParser(); p.add_argument("paths",nargs="+"); args=p.parse_args()
    runtime=build_runtime(); records=ingest_sources(args.paths); runtime["index"](records)
    print(f"Indexed {len(records)} chunks.")
if __name__=="__main__": main()
