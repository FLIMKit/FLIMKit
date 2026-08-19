import argparse
import os


def main(fast=False, cli=False, check_updates=False, no_plugins=False, plugin_path=None):
    if no_plugins:
        os.environ['FLIMKIT_NO_PLUGINS'] = '1'
    if plugin_path:
        existing = os.environ.get('FLIMKIT_PLUGIN_PATH', '')
        os.environ['FLIMKIT_PLUGIN_PATH'] = (
            existing + os.pathsep + os.pathsep.join(plugin_path) if existing
            else os.pathsep.join(plugin_path))
    if check_updates:
        from flimkit.utils.update_check import (
            check_installation_freshness,
            format_update_report,
        )
        report = format_update_report(
            check_installation_freshness(timeout=3.0, do_fetch=True)
        )
        print(report)
        return
    if not cli:
        from flimkit.UI.gui import launch_gui
        launch_gui()
        return
    from flimkit import plugins
    plugins.ensure_loaded()
    from flimkit.interactive import single_FOV_flim_fit, stitch_and_fit, stitch_tiles, timelapse_flim_fit, zstack_flim_fit
    import inquirer
    from flimkit._version import __version__, roadmap
    from flimkit.utils.fancy import display_banner, flim_fitting_banner, banner_goodbye
    if fast == False:
        display_banner()
    print('Welcome to the FLIM data processing tool!')
    questions = [
        inquirer.List(
            'process_option',
            message='Choose a processing option',
            choices=[
                'FLIM FIT a single FOV',
                'Phasor analysis',
                'Reconstruct a FOV and FLIM FIT',
                'Just stitch multiple tiles together',
                'Timelapse batch fit',
                'Z-stack batch fit',
                'About',
                'Exit'
            ]
        )
    ]
    answers = inquirer.prompt(questions)
    if answers['process_option'] == 'FLIM FIT a single FOV':
        if fast == False:
            flim_fitting_banner()
        print('FLIM FITting a single FOV...')
        single_FOV_flim_fit(interactive=True)
    elif answers['process_option'] == 'Phasor analysis':
        from flimkit.phasor_launcher import phasor_inquire
        phasor_inquire()
    elif answers['process_option'] == 'Reconstruct a FOV and FLIM FIT':
        print('Reconstructing a FOV and FLIM FITting...')
        stitch_and_fit(interactive=True)
    elif answers['process_option'] == 'Just stitch multiple tiles together':
        print('Stitching multiple tiles together...')
        stitch_tiles(interactive=True)
    elif answers['process_option'] == 'Timelapse batch fit':
        print('Timelapse batch FLIM fitting...')
        timelapse_flim_fit(interactive=True)
    elif answers['process_option'] == 'Z-stack batch fit':
        print('Z-stack batch FLIM fitting...')
        zstack_flim_fit(interactive=True)
    elif answers['process_option'] == 'About':
        print('Current version: ' + __version__)
        try:
            from flimkit.utils.update_check import (
                check_installation_freshness,
                format_update_report,
            )
            print()
            print(format_update_report(
                check_installation_freshness(timeout=2.5, do_fetch=False)
            ))
        except Exception as exc:
            print(f'Update check unavailable: {exc}')
        print(roadmap)
        return    
    else:
        banner_goodbye()
        return
    
def run(argv=None):
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='FLIMKit — FLIM data processing toolkit')
    parser.add_argument('--cli', action='store_true', help='=Run in CLI mode')
    parser.add_argument('--fast', action='store_true', help='Skip banner display')
    parser.add_argument(
        '--check-updates',
        action='store_true',
        help='Check whether git checkout and local version are up to date, then exit',
    )
    parser.add_argument(
        '--no-plugins',
        action='store_true',
        help='Skip every plugin, built in and user, for a reproducible baseline',
    )
    parser.add_argument(
        '--plugins',
        action='append',
        metavar='DIR',
        help='Extra folder to load plugins from, repeatable',
    )
    args = parser.parse_args(argv)
    main(fast=args.fast, cli=args.cli, check_updates=args.check_updates,
         no_plugins=args.no_plugins, plugin_path=args.plugins)


if __name__ == '__main__':
    run()
