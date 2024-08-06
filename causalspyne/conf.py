import argparse
import configparser
# ConfigArgParse is a drop-in replacement for argparse that allows options to
# be set via config files and environment variables.


class DynamicConfig:
    def __init__(self, conf):
        for key, value in conf.items():
            setattr(self, key, value)


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    # parser = configargparse.ArgumentParser(description='Your program description')
    parser.add_argument('-c', '--config', help='Config file path')
    parser.add_argument('--option1', help='Option 1 description')
    parser.add_argument('--option2', help='Option 2 description')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if args.config:
        config.read(args.config)

    # Merge config file and command line arguments
    options = {}
    if 'DEFAULT' in config:
        options.update(config['DEFAULT'])
    options.update({k: v for k, v in vars(args).items() if v is not None})

    return options
def load_config_from_file(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def parse(args):
    if args.config:
        config = load_config_from_file(args.config)
        db_host = config.get('database', 'host', fallback=args.db_host)
        db_port = config.getint('database', 'port', fallback=args.db_port)
        db_user = config.get('database', 'user', fallback=args.db_user)
        db_password = config.get('database', 'password', fallback=args.db_password)
        db_name = config.get('database', 'dbname', fallback=args.db_name)
        server_host = config.get('server', 'host', fallback=args.server_host)
        server_port = config.getint('server', 'port', fallback=args.server_port)
    else:
        db_host = args.db_host
        db_port = args.db_port
        db_user = args.db_user
        db_password = args.db_password
        db_name = args.db_name
        server_host = args.server_host
        server_port = args.server_port

def test_args():
    options = parse_args_and_config()
    print(options)

