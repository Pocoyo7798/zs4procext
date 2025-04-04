import json
import os
from pydantic import ValidationError
from zs4procext.parser_graphs import DataConverter
import click

@click.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_file', type=click.Path())
def main(input_file, output_file):
    """Convert data from INPUT_FILE to JSON and write to OUTPUT_FILE."""
    data_string = input_file.read()
    try:
        converter = DataConverter(data_string=data_string)
        json_data = converter.to_json()

        # Ensure the output file has a proper extension and filename
        if os.path.isdir(output_file):
            output_file = os.path.join(output_file, 'results.json')
        elif not output_file.endswith('.json'):
            output_file += '.json'

        with open(output_file, 'w') as f:
            f.write(json_data)
        click.echo(f'Converted data has been written to {output_file}')
    except ValidationError as e:
        click.echo(f'Error: {e.json()}')

if __name__ == '__main__':
    main()