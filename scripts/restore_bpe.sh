#!/bin/bash

sed -r 's/(@@ )|(@@ ?$)//g' $1 < /dev/stdin