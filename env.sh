#!/bin/sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export GC_DM_BASE=$SCRIPT_DIR
export GC_DM_DATA='/data/user/tchau/DarkMatter_OscNext/'
export GC_DM_OUTPUT='/data/user/tchau/DarkMatter_OscNext/Sensitivity'