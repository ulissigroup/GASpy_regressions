def vasp_settings_to_str(vasp_settings):
    vasp_settings=vasp_settings.copy()
    for key in vasp_settings:
        if type(vasp_settings[key]) not in [str,int,float,bool]:
            vasp_settings[key]=str(vasp_settings[key])
    return vasp_settings
