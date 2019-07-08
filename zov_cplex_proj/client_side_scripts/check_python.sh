#!/bin/bash
case "$(python --version 2>&1)" in
    *" 3."*)
        echo "Python version is Fine!"
        ;;
    *)
        echo "Parsl requires Python 3!!!"
	exit 1
        ;;
esac
