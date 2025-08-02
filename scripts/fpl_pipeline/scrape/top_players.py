from scripts.fpl_pipeline.scrape.api_client import *
from scripts.fpl_pipeline.utils.parse_helpers import *

def main():
    data = get_data()
    parse_top_players(data, 'data/2021-22')

if __name__ == '__main__':
    main()
