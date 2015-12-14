#!/bin/bash
#This script updates the submodules for FERN.
#Released under the project license.
#
#Author: Jay Jay Billings
#Author Contact: billingsjj@ornl.gov
git submodule foreach git pull origin master
