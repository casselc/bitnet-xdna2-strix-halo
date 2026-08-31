#!/usr/bin/env bb
;; CPU harness tenant for the tri-device co-tenancy measurement.
;;
;; The brief asks for a workload resembling the eventual control plane rather
;; than a synthetic integer loop, so this is structured Clojure evaluation of
;; the shape a verifier would actually do: build a small state graph, walk it,
;; check invariants over the result, and reject anything malformed. Babashka
;; (SCI) is the same interpreter family the real harness would use.
;;
;; Reports throughput and p50/p95 latency per operation so CPU headroom is
;; visible as a LATENCY distribution, not just a rate -- a control plane that
;; keeps its median but blows its tail is not healthy under co-tenancy.

(require '[clojure.string :as str])

(defn build-graph
  "A dependency graph of n nodes, each depending on up to 3 earlier ones."
  [n seed]
  (let [rng (java.util.Random. seed)]
    (into {} (for [i (range n)]
               [i (vec (distinct (for [_ (range (.nextInt rng 4))
                                       :when (pos? i)]
                                   (.nextInt rng i))))]))))

(defn topo-order
  "Kahn's algorithm over `g` (node -> its dependencies). A node is ready when
  every dependency has been emitted, so in-degree is the node's OWN dependency
  count and the reverse index says who to decrement. Returns nil on a cycle,
  which the checker treats as a rejected plan rather than an exception."
  [g]
  (let [indeg (into {} (map (fn [[n deps]] [n (count deps)]) g))
        rdeps (reduce (fn [m [n deps]]
                        (reduce #(update %1 %2 (fnil conj []) n) m deps))
                      {} g)]
    (loop [ready (into [] (keys (filter (comp zero? val) indeg)))
           indeg indeg out []]
      (if-let [n (peek ready)]
        (let [ready (pop ready)
              [indeg ready] (reduce (fn [[i r] m]
                                      (let [i (update i m dec)]
                                        [i (if (zero? (i m)) (conj r m) r)]))
                                    [indeg ready] (get rdeps n []))]
          (recur ready indeg (conj out n)))
        (when (= (count out) (count g)) out)))))

(defn invariants-hold?
  "Every dependency of a node must appear before it in the order."
  [g order]
  (let [pos (into {} (map-indexed (fn [i n] [n i]) order))]
    (every? (fn [[n deps]] (every? #(< (pos %) (pos n)) deps)) g)))

(defn one-op [i]
  (let [g (build-graph 220 i)
        order (topo-order g)]
    (and order (invariants-hold? g order))))

(let [args (into {} (map vec (partition 2 (map read-string *command-line-args*))))
      secs (get args 'secs 30)
      t0   (System/nanoTime)
      deadline (+ t0 (* secs 1e9))]
  (loop [n 0 lat (transient [])]
    (if (< (System/nanoTime) deadline)
      (let [s (System/nanoTime)
            ok (one-op n)
            e (System/nanoTime)]
        (when-not ok (binding [*out* *err*] (println "INVARIANT FAILED at" n)))
        (recur (inc n) (conj! lat (/ (- e s) 1e6))))
      (let [lat (sort (persistent! lat))
            c (count lat)
            wall (/ (- (System/nanoTime) t0) 1e9)
            pct (fn [p] (nth lat (min (dec c) (int (* p c))) 0))]
        (println (format "{\"ops\": %d, \"wall_s\": %.2f, \"ops_per_s\": %.1f, \"p50_ms\": %.3f, \"p95_ms\": %.3f}"
                         c wall (/ c wall) (double (pct 0.50)) (double (pct 0.95))))))))
