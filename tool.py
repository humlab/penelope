import click


@click.group(invoke_without_command=True)
@click.option('--output-filename', type=click.STRING)
@click.pass_context
def cli(ctx, output_filename):
    click.echo(f"output {output_filename}")
    if ctx.invoked_subcommand is None:
        click.echo('I was invoked without subcommand')
    else:
        click.echo(f"I am about to invoke {ctx.invoked_subcommand}")


@cli.command()
@click.option('--options-filename', type=click.STRING)
def sync(options_filename):
    click.echo(f'The subcommand says {options_filename}')


if __name__ == '__main__':
    cli()
