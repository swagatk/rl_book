PIDS=$(ps aux | sed -n '/chrome/s/[^0-9]*\([0-9]\+\).*/\1/p')

for PID in $PIDS; do
	if [[ -n "$PID" ]]; then
		kill -TERM "$PID"
		sleep 2
		if [[ ! -d "/proc/$PID" ]]; then
			echo "Process $PID terminated successfully"
		else
			echo "process $PID did not terminate gracefully"
			kill -KILL "$PID"
		fi
	else
		echo "process not found"
	fi
done
