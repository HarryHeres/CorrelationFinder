#!/usr/bin/env bash

cmake . -B build
cd build && cmake --build .
cd ..
