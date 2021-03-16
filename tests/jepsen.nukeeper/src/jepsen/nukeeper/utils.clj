(ns jepsen.nukeeper.utils
  (:require [clojure.string :as str]
            [zookeeper.data :as data]
            [zookeeper :as zk]
            [jepsen.control.util :as cu]
            [jepsen.nukeeper.constants :refer :all]
            [jepsen.control :as c]
            [clojure.tools.logging :refer :all]))

(defn parse-long
  "Parses a string to a Long. Passes through `nil` and empty strings."
  [s]
  (if (and s (> (count s) 0))
    (Long/parseLong s)))

(defn zk-range
  []
  (map (fn [v] (str "/" v)) (range)))

(defn zk-path
  [n]
  (str "/" n))

(defn zk-connect
  [host port timeout]
  (zk/connect (str host ":" port) :timeout-msec timeout))

(defn zk-create-range
  [conn n]
  (dorun (map (fn [v] (zk/create-all conn v :persistent? true)) (take n (zk-range)))))

(defn zk-set
  ([conn path value]
   (zk/set-data conn path (data/to-bytes (str value)) -1))
  ([conn path value version]
   (zk/set-data conn path (data/to-bytes (str value)) version)))

(defn zk-get-str
  [conn path]
  (let [zk-result (zk/data conn path)]
    {:data (data/to-string (:data zk-result))
     :stat (:stat zk-result)}))

(defn zk-list
  [conn path]
  (zk/children conn path))

(defn zk-cas
  [conn path old-value new-value]
  (let [current-value (zk-get-str conn path)]
    (if (= (parse-long (:data current-value)) old-value)
      (do (zk-set conn path new-value (:version (:stat current-value)))
          true))))

(defn zk-add-to-set
  [conn path elem]
  (let [current-value (zk-get-str conn path)
        current-set (read-string (:data current-value))
        new-set (conj current-set elem)]
    (zk-set conn path (pr-str new-set) (:version (:stat current-value)))))

(defn zk-create-if-not-exists
  [conn path data]
  (zk/create conn path :data (data/to-bytes (str data)) :persistent? true))

(defn clickhouse-alive?
  [node test]
  (info "Checking server alive on" node)
  (try
    (c/exec (str binary-path "/clickhouse") :client :--query "SELECT 1")
    (catch Exception _ false)))

(defn wait-clickhouse-alive!
  [node test & {:keys [maxtries] :or {maxtries 30}}]
  (loop [i 0]
    (cond (> i maxtries) false
          (clickhouse-alive? node test) true
          :else (do (Thread/sleep 1000) (recur (inc i))))))

(defn kill-clickhouse!
  [node test]
  (info "Killing server on node" node)
  (c/su
   (cu/stop-daemon! (str binary-path "/clickhouse") pidfile)))

(defn start-clickhouse!
  [node test]
  (info "Starting server on node" node)
  (c/su
   (cu/start-daemon!
    {:pidfile pidfile
     :logfile logfile
     :chdir dir}
    (str binary-path "/clickhouse")
    :server
    :--config "/etc/clickhouse-server/config.xml"))
  (wait-clickhouse-alive! node test))
