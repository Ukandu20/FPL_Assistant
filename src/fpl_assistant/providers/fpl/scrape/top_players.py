from fpl_assistant.providers.fpl.scrape.api_client import *
from fpl_assistant.providers.fpl.utils.parse_helpers import *

def main():
    data = get_data()
    parse_top_players(data, 'data/2021-22')

if __name__ == '__main__':
    main()
