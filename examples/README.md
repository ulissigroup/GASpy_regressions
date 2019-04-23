This folder contains some of the scripts that we run via cron jobs everyday.
They are currently hard-coded for our setup. You will have to configure them
for your own specific needs, but they will at least give you an idea of
what/how we do things.

For example:  we have this in our crontab:
    0 0 * * * /path/to/GASpy/GASpy_regressions/examples/submit_slurm_jobs.sh
    0 18 * * * /path/to/GASpy/GASpy_regressions/examples/push_predictions.sh
