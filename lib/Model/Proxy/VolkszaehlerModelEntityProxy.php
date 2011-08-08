<?php

namespace Volkszaehler\Model\Proxy;

/**
 * THIS CLASS WAS GENERATED BY THE DOCTRINE ORM. DO NOT EDIT THIS FILE.
 */
class VolkszaehlerModelEntityProxy extends \Volkszaehler\Model\Entity implements \Doctrine\ORM\Proxy\Proxy
{
    private $_entityPersister;
    private $_identifier;
    public $__isInitialized__ = false;
    public function __construct($entityPersister, $identifier)
    {
        $this->_entityPersister = $entityPersister;
        $this->_identifier = $identifier;
    }
    /** @private */
    public function __load()
    {
        if (!$this->__isInitialized__ && $this->_entityPersister) {
            $this->__isInitialized__ = true;
            if ($this->_entityPersister->load($this->_identifier, $this) === null) {
                throw new \Doctrine\ORM\EntityNotFoundException();
            }
            unset($this->_entityPersister, $this->_identifier);
        }
    }
    
    
    public function checkProperties()
    {
        $this->__load();
        return parent::checkProperties();
    }

    public function getProperty($key)
    {
        $this->__load();
        return parent::getProperty($key);
    }

    public function getProperties($prefix = NULL)
    {
        $this->__load();
        return parent::getProperties($prefix);
    }

    public function setProperty($key, $value)
    {
        $this->__load();
        return parent::setProperty($key, $value);
    }

    public function deleteProperty($key)
    {
        $this->__load();
        return parent::deleteProperty($key);
    }

    public function getId()
    {
        $this->__load();
        return parent::getId();
    }

    public function getUuid()
    {
        $this->__load();
        return parent::getUuid();
    }

    public function getType()
    {
        $this->__load();
        return parent::getType();
    }

    public function getDefinition()
    {
        $this->__load();
        return parent::getDefinition();
    }


    public function __sleep()
    {
        return array('__isInitialized__', 'id', 'uuid', 'type', 'properties', 'parents');
    }

    public function __clone()
    {
        if (!$this->__isInitialized__ && $this->_entityPersister) {
            $this->__isInitialized__ = true;
            $class = $this->_entityPersister->getClassMetadata();
            $original = $this->_entityPersister->load($this->_identifier);
            if ($original === null) {
                throw new \Doctrine\ORM\EntityNotFoundException();
            }
            foreach ($class->reflFields AS $field => $reflProperty) {
                $reflProperty->setValue($this, $reflProperty->getValue($original));
            }
            unset($this->_entityPersister, $this->_identifier);
        }
        
    }
}