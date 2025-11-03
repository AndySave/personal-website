"use client";

import { NetworkMetadata } from "@/types/nn-framework";
import { useState, useEffect } from "react";

export default function useNetworks() {
  const [selectedNetwork, setSelectedNetwork] =
    useState<NetworkMetadata | null>(null);
  const [networks, setNetworks] = useState<NetworkMetadata[] | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL;

  useEffect(() => {
    fetch(`${API_URL}/api/nn-framework/networks-metadata`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Server error ${res.status}`);
        }
        return res.json();
      })
      .then((data: NetworkMetadata[]) => {
        setNetworks(data);
        setSelectedNetwork(data[0]);
      })
      .catch((err) => console.error(err));
  }, []);

  useEffect(() => {
    // TODO: Temp for debug
    if (networks) {
      console.log("Networks fetched:", networks);
    }
  }, [networks]);

  return {
    networks,
    selectedNetwork,
    setSelectedNetwork,
  };
}
